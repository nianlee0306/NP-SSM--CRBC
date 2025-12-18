import os, time, random, math, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_DIR, DATASET_NAME, CKPT_DIR, LOG_FILE, RNG_SEED,
    MAX_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, CLIP_NORM,
    VAL_SPLIT, AMP, EARLY_STOP_PATIENCE, LAMBDA_EDGE, LAMBDA_CORNER,
    DEVICE, INPUT_NOISE_STD
)
from model import NP_SSM_CRBC

TEST_SPLIT = 0.1
MIN_VAL_EP = 8
MIN_TEST_EP = 8

DEVICE = DEVICE
AMP_ENABLED = bool(AMP and (DEVICE == "cuda"))
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def log(s: str):
    print(s)
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fmt(v, nd=10):
    if isinstance(v, (float, int)):
        return f"{v:.{nd}f}"
    return str(v)


def current_lr(optimizer: torch.optim.Optimizer):
    if not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0].get("lr", 0.0)


def grad_global_norm(model: nn.Module):
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sq_sum += float(torch.sum(g * g).item())
    return math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0


class CRBCDataset(Dataset):
    def __init__(self, split: str, mu_override=None, std_override=None):
        npz = np.load(DATA_DIR / f"{DATASET_NAME}.npz")
        self.X = npz["X"].astype(np.float32)  # 未归一化
        self.Y = npz["Y"].astype(np.float32)
        self.mask_edge = npz["mask_edge"].astype(bool)
        self.mask_corner = npz["mask_corner"].astype(bool)
        self.ep_ids = npz["ep_ids"].astype(np.int32)
        self.T = int(npz["T"])

        ep_all = np.unique(self.ep_ids)
        rng = np.random.RandomState(RNG_SEED)
        rng.shuffle(ep_all)

        n_val_ep = max(MIN_VAL_EP, int(len(ep_all) * VAL_SPLIT))
        n_test_ep = max(MIN_TEST_EP, int(len(ep_all) * TEST_SPLIT))

        # [val][test][train]（按 episode 划分）
        ep_val = set(ep_all[:n_val_ep])
        ep_test = set(ep_all[n_val_ep:n_val_ep + n_test_ep])
        ep_tr = set(ep_all[n_val_ep + n_test_ep:])

        if split == "train":
            sel = np.where(np.isin(self.ep_ids, list(ep_tr)))[0]
        elif split == "val":
            sel = np.where(np.isin(self.ep_ids, list(ep_val)))[0]
        elif split == "test":
            sel = np.where(np.isin(self.ep_ids, list(ep_test)))[0]
        else:
            raise ValueError(f"Unknown split={split}")
        self.sel = sel
        '''
        n_val_ep = max(8, int(len(ep_all) * VAL_SPLIT))
        ep_val = set(ep_all[:n_val_ep])
        ep_tr  = set(ep_all[n_val_ep:])

        if split == "train":
            sel = np.where(np.isin(self.ep_ids, list(ep_tr)))[0]
        else:
            sel = np.where(np.isin(self.ep_ids, list(ep_val)))[0]
        self.sel = sel
'''
        if split == "train" and (mu_override is None or std_override is None):
            Xflat = self.X[self.sel].reshape(-1, self.X.shape[-1])
            self.mu = Xflat.mean(axis=0).astype(np.float32)
            self.std = (Xflat.std(axis=0) + 1e-6).astype(np.float32)
        else:
            assert mu_override is not None and std_override is not None, \
                "val split 需要指定 train 的 mu/std"
            self.mu = mu_override.astype(np.float32)
            self.std = std_override.astype(np.float32)

    def __len__(self):
        return self.sel.shape[0]

    def __getitem__(self, i):
        j = self.sel[i]
        X = (self.X[j] - self.mu) / self.std
        Y = self.Y[j]
        me = bool(self.mask_edge[j])
        mc = bool(self.mask_corner[j])
        return (
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.tensor(me, dtype=torch.bool),
            torch.tensor(mc, dtype=torch.bool),
        )


@torch.no_grad()
def _mae(a: torch.Tensor, b: torch.Tensor):
    return torch.mean(torch.abs(a - b))


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k] * (1.0 - self.decay))
            else:
                self.ema.state_dict()[k].copy_(msd[k])


def run_one_epoch(model, loader, optimizer=None, scaler=None, ema: ModelEma = None, add_input_noise=False):
    is_train = optimizer is not None
    model.train(is_train)

    l1loss = nn.SmoothL1Loss(reduction="none", beta=0.01)

    sum_loss = 0.0
    sum_edge_loss = 0.0
    sum_corner_loss = 0.0
    sum_edge_mae = 0.0
    sum_corner_mae = 0.0

    n_samples = 0
    n_edge_eff = 0
    n_corner_eff = 0
    n_batches = 0

    grad_norm_max = 0.0
    grad_norm_sum = 0.0

    t0 = time.time()

    for X, Y, mask_e, mask_c in loader:
        n_batches += 1
        B, T, _ = X.shape
        n_samples += B

        X, Y, mask_e, mask_c = (
            X.to(DEVICE, non_blocking=True),
            Y.to(DEVICE, non_blocking=True),
            mask_e.to(DEVICE, non_blocking=True),
            mask_c.to(DEVICE, non_blocking=True),
        )
        if is_train and add_input_noise and INPUT_NOISE_STD > 0:
            X = X + torch.randn_like(X) * INPUT_NOISE_STD

        state = model.init_state(B, device=DEVICE)

        def fwd(return_outputs_for_metrics=False):
            nonlocal state
            loss_e, loss_c = 0.0, 0.0
            mae_e, mae_c = 0.0, 0.0

            for t in range(T):
                o1, o3, state = model(X[:, t, :], state)

                if mask_e.any():
                    l1 = l1loss(o1, Y[:, t, 0:1]).squeeze(-1)[mask_e]
                    if l1.numel() > 0:
                        loss_e = loss_e + l1.mean()
                        mae_e = mae_e + torch.mean(torch.abs(o1[mask_e] - Y[:, t, 0:1][mask_e]))

                if mask_c.any():
                    l3 = l1loss(o3, Y[:, t, 0:3]).mean(dim=-1)[mask_c]
                    if l3.numel() > 0:
                        loss_c = loss_c + l3.mean()
                        mae_c = mae_c + _mae(o3[mask_c], Y[:, t, 0:3][mask_c])

            steps = max(1, T)
            loss_e = loss_e / steps if isinstance(loss_e, torch.Tensor) else torch.tensor(0.0, device=DEVICE)
            loss_c = loss_c / steps if isinstance(loss_c, torch.Tensor) else torch.tensor(0.0, device=DEVICE)
            total = LAMBDA_EDGE * loss_e + LAMBDA_CORNER * loss_c

            if return_outputs_for_metrics:
                return total, loss_e, loss_c, (
                    mae_e / steps if isinstance(mae_e, torch.Tensor) else torch.tensor(0.0, device=DEVICE)), (
                           mae_c / steps if isinstance(mae_c, torch.Tensor) else torch.tensor(0.0, device=DEVICE))
            return total

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if AMP_ENABLED:
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    loss, loss_e, loss_c, mae_e, mae_c = fwd(return_outputs_for_metrics=True)
            else:
                loss, loss_e, loss_c, mae_e, mae_c = fwd(return_outputs_for_metrics=True)

            if AMP_ENABLED:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)

            if AMP_ENABLED:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if ema is not None:
                ema.update(model)

            gn = grad_global_norm(model)
            grad_norm_sum += gn
            grad_norm_max = max(grad_norm_max, gn)
        else:
            with torch.no_grad():
                loss, loss_e, loss_c, mae_e, mae_c = fwd(return_outputs_for_metrics=True)

        sum_loss += float(loss.item()) * B
        sum_edge_loss += float(loss_e.item()) * B
        sum_corner_loss += float(loss_c.item()) * B
        sum_edge_mae += float(mae_e.item()) * B
        sum_corner_mae += float(mae_c.item()) * B
        n_edge_eff += int(mask_e.sum().item())
        n_corner_eff += int(mask_c.sum().item())

    elapse = time.time() - t0
    avg = lambda s: (s / max(1, n_samples))
    metrics = {
        "loss": avg(sum_loss),
        "edge_loss": avg(sum_edge_loss),
        "corner_loss": avg(sum_corner_loss),
        "edge_mae": avg(sum_edge_mae),
        "corner_mae": avg(sum_corner_mae),
        "n_edge_eff": int(n_edge_eff),
        "n_corner_eff": int(n_corner_eff),
        "n_samples": int(n_samples),
        "n_batches": int(n_batches),
        "time_sec": elapse,
        "throughput_sps": n_samples / max(1e-9, elapse),
    }
    if is_train:
        metrics.update({
            "grad_norm_avg": grad_norm_sum / max(1, n_batches),
            "grad_norm_max": grad_norm_max,
        })
    return metrics


import matplotlib.pyplot as plt


@torch.no_grad()
def eval_collect(model, loader, max_points=200000):
    model.eval()

    l1loss = nn.SmoothL1Loss(reduction="none", beta=0.01)

    sum_loss = 0.0
    sum_edge_loss = 0.0
    sum_corner_loss = 0.0
    sum_edge_mae = 0.0
    sum_corner_mae = 0.0
    n_samples = 0
    n_edge_eff = 0
    n_corner_eff = 0
    n_batches = 0

    # 收集用于画图的数据（按点采样，防止爆内存）
    y_edge_pred, y_edge_true = [], []
    y_corner_pred, y_corner_true = [], []

    for X, Y, mask_e, mask_c in loader:
        n_batches += 1
        B, T, _ = X.shape
        n_samples += B

        X, Y, mask_e, mask_c = (
            X.to(DEVICE, non_blocking=True),
            Y.to(DEVICE, non_blocking=True),
            mask_e.to(DEVICE, non_blocking=True),
            mask_c.to(DEVICE, non_blocking=True),
        )

        state = model.init_state(B, device=DEVICE)

        loss_e, loss_c = 0.0, 0.0
        mae_e, mae_c = 0.0, 0.0

        for t in range(T):
            o1, o3, state = model(X[:, t, :], state)

            if mask_e.any():
                l1 = l1loss(o1, Y[:, t, 0:1]).squeeze(-1)[mask_e]
                if l1.numel() > 0:
                    loss_e = loss_e + l1.mean()
                    mae_e = mae_e + torch.mean(torch.abs(o1[mask_e] - Y[:, t, 0:1][mask_e]))

                    if len(y_edge_pred) < max_points:
                        y_edge_pred.append(o1[mask_e].detach().cpu().numpy())
                        y_edge_true.append(Y[:, t, 0:1][mask_e].detach().cpu().numpy())

            if mask_c.any():
                l3 = l1loss(o3, Y[:, t, 0:3]).mean(dim=-1)[mask_c]
                if l3.numel() > 0:
                    loss_c = loss_c + l3.mean()
                    mae_c = mae_c + _mae(o3[mask_c], Y[:, t, 0:3][mask_c])

                    if len(y_corner_pred) < max_points:
                        y_corner_pred.append(o3[mask_c].detach().cpu().numpy())
                        y_corner_true.append(Y[:, t, 0:3][mask_c].detach().cpu().numpy())

        steps = max(1, T)
        loss_e = loss_e / steps if isinstance(loss_e, torch.Tensor) else torch.tensor(0.0, device=DEVICE)
        loss_c = loss_c / steps if isinstance(loss_c, torch.Tensor) else torch.tensor(0.0, device=DEVICE)
        total = LAMBDA_EDGE * loss_e + LAMBDA_CORNER * loss_c

        sum_loss += float(total.item()) * B
        sum_edge_loss += float(loss_e.item()) * B
        sum_corner_loss += float(loss_c.item()) * B
        sum_edge_mae += float((mae_e / steps).item()) * B if isinstance(mae_e, torch.Tensor) else 0.0
        sum_corner_mae += float((mae_c / steps).item()) * B if isinstance(mae_c, torch.Tensor) else 0.0

        n_edge_eff += int(mask_e.sum().item())
        n_corner_eff += int(mask_c.sum().item())

    avg = lambda s: (s / max(1, n_samples))
    metrics = {
        "loss": avg(sum_loss),
        "edge_loss": avg(sum_edge_loss),
        "corner_loss": avg(sum_corner_loss),
        "edge_mae": avg(sum_edge_mae),
        "corner_mae": avg(sum_corner_mae),
        "n_edge_eff": int(n_edge_eff),
        "n_corner_eff": int(n_corner_eff),
        "n_samples": int(n_samples),
        "n_batches": int(n_batches),
    }

    def _cat_or_empty(xs):
        if not xs:
            return np.zeros((0,), dtype=np.float32)
        arr = np.concatenate(xs, axis=0)
        return arr

    edge_pred = _cat_or_empty(y_edge_pred).reshape(-1)
    edge_true = _cat_or_empty(y_edge_true).reshape(-1)

    corner_pred = _cat_or_empty(y_corner_pred).reshape(-1, 3)
    corner_true = _cat_or_empty(y_corner_true).reshape(-1, 3)

    return metrics, (edge_pred, edge_true, corner_pred, corner_true)


def save_learning_curves(hist, out_path):
    x = hist["epoch"]

    plt.figure()
    plt.plot(x, hist["train_loss"], label="train")
    plt.plot(x, hist["val_loss"], label="val")
    plt.plot(x, hist["test_loss"], label="test")
    plt.xlabel("epoch");
    plt.ylabel("loss");
    plt.legend()
    plt.tight_layout();
    plt.savefig(out_path);
    plt.close()


def save_parity(y_true, y_pred, out_path, title):
    plt.figure()
    plt.scatter(y_true, y_pred, s=3, alpha=0.4)
    mn = float(min(y_true.min(), y_pred.min())) if y_true.size else 0.0
    mx = float(max(y_true.max(), y_pred.max())) if y_true.size else 1.0
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("true");
    plt.ylabel("pred");
    plt.title(title)
    plt.tight_layout();
    plt.savefig(out_path);
    plt.close()


def save_error_hist(err, out_path, title):
    plt.figure()
    plt.hist(err, bins=80)
    plt.xlabel("pred - true");
    plt.ylabel("count");
    plt.title(title)
    plt.tight_layout();
    plt.savefig(out_path);
    plt.close()


hist = {
    "epoch": [],
    "train_loss": [], "val_loss": [], "test_loss": [],
    "train_edge_mae": [], "val_edge_mae": [], "test_edge_mae": [],
    "train_corner_mae": [], "val_corner_mae": [], "test_corner_mae": [],
}


def main():
    seed_all(RNG_SEED)
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(LOG_FILE)).mkdir(parents=True, exist_ok=True)
    PLOT_DIR = Path(CKPT_DIR).parent / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    ds_tr_tmp = CRBCDataset("train")
    mu_tr, std_tr = ds_tr_tmp.mu, ds_tr_tmp.std

    ds_tr = CRBCDataset("train", mu_override=mu_tr, std_override=std_tr)
    ds_va = CRBCDataset("val", mu_override=mu_tr, std_override=std_tr)
    ds_te = CRBCDataset("test", mu_override=mu_tr, std_override=std_tr)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    '''
    ds_tr_tmp = CRBCDataset("train")
    mu_tr, std_tr = ds_tr_tmp.mu, ds_tr_tmp.std
    ds_tr = CRBCDataset("train", mu_override=mu_tr, std_override=std_tr)
    ds_va = CRBCDataset("val",   mu_override=mu_tr, std_override=std_tr)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    '''
    model = NP_SSM_CRBC().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    #scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=AMP_ENABLED)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=LR * 1e-2)
    ema = ModelEma(model, decay=0.999)

    best_path = Path(CKPT_DIR) / "best.pth"
    last_path = Path(CKPT_DIR) / "last.pth"

    start, best_val = 1, float("inf")
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        old_base_lr = ckpt.get("base_lr", LR)
        if abs(old_base_lr - LR) > 1e-9:
            scale_factor = LR / old_base_lr
            log(f"💡 检测到基础学习率变化: {old_base_lr:.2e} -> {LR:.2e} (缩放比例: {scale_factor:.2f})")

            for param_group in opt.param_groups:
                param_group['lr'] *= scale_factor

            scheduler.base_lrs = [base_lr * scale_factor for base_lr in scheduler.base_lrs]
            log(f"   调整后当前学习率约为: {current_lr(opt):.2e}")

        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        start = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        log(f"🔁 已从 last.pth 恢复，起始 epoch = {start}")

    elif best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
        log("已载入最优模型 best.pth。")

    # log(f"设备: {DEVICE} | AMP: {AMP_ENABLED} | 训练样本: {len(ds_tr)} | 验证样本: {len(ds_va)} | T={ds_tr.T}")
    log(f"设备: {DEVICE} | AMP: {AMP_ENABLED} | 训练样本: {len(ds_tr)} | 验证样本: {len(ds_va)} | 测试样本: {len(ds_te)} | T={ds_tr.T}")
    log(f"优化器: AdamW(lr={LR}, wd={WEIGHT_DECAY}) | 批大小: {BATCH_SIZE} | 梯度裁剪: {CLIP_NORM}")

    no_imp = 0
    for epoch in range(start, MAX_EPOCHS + 1):
        ep_t0 = time.time()
        '''
        trm = run_one_epoch(model, dl_tr, opt, scaler, ema, add_input_noise=True)
        with torch.no_grad():
            ema.ema.eval()
            vam = run_one_epoch(ema.ema, dl_va)
        '''
        trm = run_one_epoch(model, dl_tr, opt, scaler, ema, add_input_noise=True)

        with torch.no_grad():
            ema.ema.eval()
            vam = run_one_epoch(ema.ema, dl_va)
            tem = run_one_epoch(ema.ema, dl_te)

        # 记录 learning curves
        hist["epoch"].append(epoch)
        hist["train_loss"].append(trm["loss"])
        hist["val_loss"].append(vam["loss"])
        hist["test_loss"].append(tem["loss"])

        hist["train_edge_mae"].append(trm["edge_mae"])
        hist["val_edge_mae"].append(vam["edge_mae"])
        hist["test_edge_mae"].append(tem["edge_mae"])

        hist["train_corner_mae"].append(trm["corner_mae"])
        hist["val_corner_mae"].append(vam["corner_mae"])
        hist["test_corner_mae"].append(tem["corner_mae"])

        scheduler.step()

        ep_t1 = time.time()
        lr_now = current_lr(opt)

        log(f"Epoch {epoch:03d} | 总耗时={fmt(ep_t1 - ep_t0, 1)}s | 学习率={fmt(lr_now, 6)}")
        log(
            "  训练 | "
            f"Loss={fmt(trm['loss'])} | 边缘Loss={fmt(trm['edge_loss'])} | 拐角Loss={fmt(trm['corner_loss'])} | "
            f"边缘MAE={fmt(trm['edge_mae'])} | 拐角MAE={fmt(trm['corner_mae'])} | "
            f"有效样本: 边缘={trm['n_edge_eff']}, 拐角={trm['n_corner_eff']} | "
            f"批次={trm['n_batches']} | 吞吐={fmt(trm['throughput_sps'], 1)} samp/s | "
            f"梯度范数: 平均={fmt(trm.get('grad_norm_avg', 0.0), 4)}, 最大={fmt(trm.get('grad_norm_max', 0.0), 4)}"
        )
        log(
            "  验证(EMA) | "
            f"Loss={fmt(vam['loss'])} | 边缘Loss={fmt(vam['edge_loss'])} | 拐角Loss={fmt(vam['corner_loss'])} | "
            f"边缘MAE={fmt(vam['edge_mae'])} | 拐角MAE={fmt(vam['corner_mae'])} | "
            f"有效样本: 边缘={vam['n_edge_eff']}, 拐角={vam['n_corner_eff']} | "
            f"批次={vam['n_batches']} | 吞吐={fmt(vam['throughput_sps'], 1)} samp/s"
        )
        log(
            "  测试(EMA) | "
            f"Loss={fmt(tem['loss'])} | 边缘Loss={fmt(tem['edge_loss'])} | 拐角Loss={fmt(tem['corner_loss'])} | "
            f"边缘MAE={fmt(tem['edge_mae'])} | 拐角MAE={fmt(tem['corner_mae'])} | "
            f"有效样本: 边缘={tem['n_edge_eff']}, 拐角={tem['n_corner_eff']} | "
            f"批次={tem['n_batches']} | 吞吐={fmt(tem['throughput_sps'], 1)} samp/s"
        )

        torch.save(
            dict(model=model.state_dict(), opt=opt.state_dict(),
                 scaler=scaler.state_dict(), epoch=epoch, best_val=best_val,
                 ema=ema.ema.state_dict(),
                 scheduler=scheduler.state_dict(),
                 base_lr=LR),
            last_path,
        )

        if vam["loss"] < best_val - 1e-12:
            best_val = vam["loss"]
            torch.save(
                dict(model=ema.ema.state_dict(), best_val=best_val, epoch=epoch, ema=ema.ema.state_dict()),
                best_path,
            )
            log(f"刷新最优：best_val={fmt(best_val)}，已保存 best.pth")
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE:
                log(f"⏹ 早停：{EARLY_STOP_PATIENCE} 个 epoch 无提升。")
                break

    # ====== 用 best.pth 在 test 上生成最终图和指标 ======
    best_ckpt = torch.load(best_path, map_location=DEVICE)
    ema.ema.load_state_dict(best_ckpt["model"])
    ema.ema.eval()

    test_metrics, (edge_pred, edge_true, corner_pred, corner_true) = eval_collect(ema.ema, dl_te)

    log("======== Test (best.pth, EMA) ========")
    log(f"  Loss={fmt(test_metrics['loss'])} | edge_MAE={fmt(test_metrics['edge_mae'])} | corner_MAE={fmt(test_metrics['corner_mae'])}")

    log("======== 训练结束 ========")

    save_learning_curves(hist, PLOT_DIR / "learning_curves_loss.png")

    if edge_true.size:
        save_parity(edge_true, edge_pred, PLOT_DIR / "parity_edge.png", "Edge parity")
        save_error_hist(edge_pred - edge_true, PLOT_DIR / "error_hist_edge.png", "Edge error histogram")

    corner_true_flat = corner_true.reshape(-1)
    corner_pred_flat = corner_pred.reshape(-1)
    if corner_true_flat.size:
        save_parity(corner_true_flat, corner_pred_flat, PLOT_DIR / "parity_corner_all3.png",
                    "Corner parity (all 3 comps)")
        save_error_hist(corner_pred_flat - corner_true_flat, PLOT_DIR / "error_hist_corner_all3.png",
                        "Corner error histogram (all 3 comps)")


if __name__ == "__main__":
    main()
