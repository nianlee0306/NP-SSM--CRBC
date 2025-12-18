import json, random, math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import meep as mp
from config import (
    DATA_DIR, RNG_SEED, NX, NY, RESOLUTION, DPML_PX, T_STEPS,
    N_EPISODES, MAX_SAMPLES_PER_EP, F_MIN, F_MAX, AMP_MIN, AMP_MAX,
    FWIDTH_MIN_RATIO, FWIDTH_MAX_RATIO, CLIP_VALUE, DATASET_NAME,
    T_WARMUP, SAMPLE_EVERY_MIN, SAMPLE_EVERY_MAX, PATCH_RADIUS
)

# ---------- 额外内部常量（可按需挪到 config.py） ----------
# 裁剪窗个数范围（每个 episode ）
NCROPS_MIN, NCROPS_MAX = 2, 5
# 旋转统一 NORTH 的边缘特征保持与旧版一致的横向保边距（防角/贴边伪影）
KEEP_MARGIN = 2
# 为防极限情况下 CFL 保守估计不够，再加一层长度裕度（单位：仿真长度单位）
SAFETY_PAD_LEN = 0.5
# 角点样本重复倍数（轻度过采样角点）
CORNER_REP = 2
# 角点样本重复倍数（轻度过采样角点）
CORNER_REP = 2

# 有多少比例的 episode 使用 “internal_source-like” 场景
# 比如 0.3 表示 30% 的 episode 用类似 comparison 脚本 internal_source 的几何 & 内部点源
INTERNAL_SCENE_FRACTION = 0.3

# internal_source-like 场景的轻微随机扰动幅度
INTERNAL_SCENE_POS_JITTER = 0.05   # 位置 jitters，相对 cell 尺度的比例
INTERNAL_SCENE_EPS_JITTER = 0.25   # 介电常数的相对扰动
INTERNAL_SCENE_FREQ_JITTER = 0.10  # 频率和 fwidth 的相对扰动

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
mp.verbosity(0)


def rot_fields_tmz(ex, ey, hz, k_rot: int):
    """在 yx 方向数组上旋转（np.rot90）并处理 TMz 的符号变换。
       注意：此版本假定输入已为 (ny, nx)（行= y，列= x）。"""
    k = k_rot % 4
    if k == 0: return ex, ey, hz
    if k == 1: return -np.rot90(ey, 1),  np.rot90(ex, 1),  np.rot90(hz, 1)
    if k == 2: return -np.rot90(ex, 2), -np.rot90(ey, 2),  np.rot90(hz, 2)
    return       np.rot90(ey, 3), -np.rot90(ex, 3),  np.rot90(hz, 3)

def decompose_tangent_normal(ex, ey):
    return ex, ey

def _as_yx(a, nx):
    return a.T

def _phys_derivs_from_patch(ex, ey, hz, y_top, y_bot, x_mid, dx):
    dhz_dx = (hz[y_bot, x_mid+1] - hz[y_bot, x_mid-1]) / (2.0 * dx)
    dhz_dy = np.mean([(hz[y_top, x_mid+k] - hz[y_bot, x_mid+k]) / dx for k in (-2,-1,0,1,2)])
    dex_dy = np.mean([(ex[y_top, x_mid+k] - ex[y_bot, x_mid+k]) / dx for k in (-2,-1,0,1,2)])
    dey_dx = np.mean([(ey[y_bot+j, x_mid+1] - ey[y_bot+j, x_mid-1]) / (2.0 * dx) for j in (0, 1)])
    curl_e = dey_dx - dex_dy
    return np.array([dhz_dx, dhz_dy, dex_dy, dey_dx, curl_e], dtype=np.float32)

def _pos_features_along_edge(x, x_start, x_end_inclusive, width_nx):
    denom = max(1.0, float(x_end_inclusive - x_start))
    u = (x - x_start) / denom
    d_left  = (x - x_start) / max(1.0, width_nx)
    d_right = (x_end_inclusive - x) / max(1.0, width_nx)
    return np.array([u, d_left, d_right], dtype=np.float32)


def _rand_gaussian():
    f0 = random.uniform(F_MIN, F_MAX)
    fwidth = random.uniform(FWIDTH_MIN_RATIO, FWIDTH_MAX_RATIO) * f0
    amp = random.uniform(AMP_MIN, AMP_MAX)
    return f0, fwidth, amp

def random_sources_in_rect(rect_phys, res):
    """
    在 rect_phys（物理坐标，中心在 (0,0) 的大域坐标系）内放置点源/短线源/短线边源。
    rect_phys = (xmin, xmax, ymin, ymax) [单位：仿真长度单位]
    """
    xmin, xmax, ymin, ymax = rect_phys
    def rand_xy(margin=0.0):
        return (random.uniform(xmin+margin, xmax-margin),
                random.uniform(ymin+margin, ymax-margin))
    srcs = []

    if random.random() < 0.85:
        n = random.randint(1, 4)
        margin = 4.0 / res
        for _ in range(n):
            x, y = rand_xy(margin)
            f0, fwidth, amp = _rand_gaussian()
            comp = random.choice([mp.Hz, mp.Ex, mp.Ey])
            if random.random() < 0.7:
                srcs.append(mp.Source(src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
                                      component=comp, center=mp.Vector3(x, y),
                                      amplitude=amp*random.uniform(0.7, 1.3)))
            else:
                lx = random.uniform(0.0, 0.15*(xmax-xmin))
                ly = random.uniform(0.0, 0.15*(ymax-ymin))
                srcs.append(mp.Source(src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
                                      component=comp, center=mp.Vector3(x, y),
                                      size=mp.Vector3(lx, ly),
                                      amplitude=amp*random.uniform(0.6, 1.4)))

    if random.random() < 0.35:
        f0, fwidth, amp = _rand_gaussian()
        which = random.choice(["N","S","E","W"])
        shrink = 0.2
        if which in ("N","S"):
            y = ymax - 2.0/res if which=="N" else ymin + 2.0/res
            span = (xmax-xmin)*(1.0 - shrink)
            srcs.append(mp.Source(src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
                                  component=random.choice([mp.Hz, mp.Ex]),
                                  center=mp.Vector3(0.5*(xmin+xmax), y),
                                  size=mp.Vector3(span, 0),
                                  amplitude=amp))
        else:
            x = xmax - 2.0/res if which=="E" else xmin + 2.0/res
            span = (ymax-ymin)*(1.0 - shrink)
            srcs.append(mp.Source(src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
                                  component=random.choice([mp.Hz, mp.Ey]),
                                  center=mp.Vector3(x, 0.5*(ymin+ymax)),
                                  size=mp.Vector3(0, span),
                                  amplitude=amp))
    return srcs

def random_geometry_in_rect(rect_phys, res):
    """仅在裁剪窗内随机几何（外侧真空），避免贴边过近。"""
    xmin, xmax, ymin, ymax = rect_phys
    geos = []
    if random.random() < 0.75:
        n_obj = random.randint(1, 4)
        margin = 3.0 / res
        for _ in range(n_obj):
            if random.random() < 0.55:
                sx = random.uniform(0.08*(xmax-xmin), 0.28*(xmax-xmin))
                sy = random.uniform(0.08*(ymax-ymin), 0.28*(ymax-ymin))
                cx = random.uniform(xmin+margin+sx/2, xmax-margin-sx/2)
                cy = random.uniform(ymin+margin+sy/2, ymax-margin-sy/2)
                eps = random.uniform(1.2, 12.0)
                if random.random() < 0.5:
                    theta = random.uniform(0, math.pi)
                    e1 = mp.Vector3(math.cos(theta), math.sin(theta))
                    e2 = mp.Vector3(-math.sin(theta), math.cos(theta))
                    geos.append(mp.Block(center=mp.Vector3(cx, cy),
                                         size=mp.Vector3(sx, sy),
                                         e1=e1, e2=e2,
                                         material=mp.Medium(epsilon=eps)))
                else:
                    geos.append(mp.Block(center=mp.Vector3(cx, cy),
                                         size=mp.Vector3(sx, sy),
                                         material=mp.Medium(epsilon=eps)))
            else:
                r = random.uniform(0.05*min(xmax-xmin, ymax-ymin),
                                   0.18*min(xmax-xmin, ymax-ymin))
                cx = random.uniform(xmin+margin+r, xmax-margin-r)
                cy = random.uniform(ymin+margin+r, ymax-margin-r)
                eps = random.uniform(1.2, 12.0)
                geos.append(mp.Cylinder(center=mp.Vector3(cx, cy), radius=r,
                                        height=mp.inf, material=mp.Medium(epsilon=eps)))
    return geos

def make_internal_source_like_scene(rect_phys, res):
    """
    生成一个类似 comparison 脚本 define_scene_internal_source 的场景：
    - 一个偏右上方的高 ε 方块
    - 一个偏左下的圆柱
    - 中心略偏右上的内部 Hz 点源

    为避免过拟合到单一场景，这里对几何位置 / 介电常数 / 频率做轻微随机扰动。
    所有坐标仅依赖裁剪窗 rect_phys 的物理尺寸（与 BASE_CELL 一致），
    这样就和测试时 internal_source 的物理分布尽量对齐。
    """
    xmin, xmax, ymin, ymax = rect_phys
    Lx = xmax - xmin
    Ly = ymax - ymin

    cell_x = Lx
    cell_y = Ly

    geos = []

    cx_b = (0.2 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_x
    cy_b = (0.25 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_y
    sx_b = 0.3 * cell_x
    sy_b = 0.3 * cell_y

    eps_b = 8.0 * (1.0 + random.uniform(-INTERNAL_SCENE_EPS_JITTER, INTERNAL_SCENE_EPS_JITTER))
    eps_b = max(1.2, eps_b) 
    geos.append(
        mp.Block(
            center=mp.Vector3(cx_b, cy_b),
            size=mp.Vector3(sx_b, sy_b),
            material=mp.Medium(epsilon=eps_b),
        )
    )

    cx_c = (-0.2 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_x
    cy_c = (-0.15 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_y
    r_c = 0.15 * min(cell_x, cell_y)

    eps_c = 4.0 * (1.0 + random.uniform(-INTERNAL_SCENE_EPS_JITTER, INTERNAL_SCENE_EPS_JITTER))
    eps_c = max(1.2, eps_c)

    geos.append(
        mp.Cylinder(
            center=mp.Vector3(cx_c, cy_c),
            radius=r_c,
            height=mp.inf,
            material=mp.Medium(epsilon=eps_c),
        )
    )

    f0_base, fwidth_base = 0.4, 0.15
    f0 = f0_base * (1.0 + random.uniform(-INTERNAL_SCENE_FREQ_JITTER, INTERNAL_SCENE_FREQ_JITTER))
    fwidth = fwidth_base * (1.0 + random.uniform(-INTERNAL_SCENE_FREQ_JITTER, INTERNAL_SCENE_FREQ_JITTER))

    amp = random.uniform(AMP_MIN, AMP_MAX)

    cx_s = (0.1 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_x
    cy_s = (0.05 + random.uniform(-INTERNAL_SCENE_POS_JITTER, INTERNAL_SCENE_POS_JITTER)) * cell_y

    srcs = [
        mp.Source(
            src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
            component=mp.Hz,
            center=mp.Vector3(cx_s, cy_s),
            amplitude=amp,
        )
    ]

    return geos, srcs

def worst_case_travel_length(res, t_warmup, t_steps, sample_every_max):
    """以 CFL 上界 dt = 1/(sqrt(2)*res) 估算：在最多步数内波可传播的最大距离。"""
    steps = t_warmup + t_steps * sample_every_max
    dt = 1.0 / (math.sqrt(2.0) * res)
    return steps * dt

def required_min_margin_units(res):
    Lmax = worst_case_travel_length(res, T_WARMUP, T_STEPS, SAMPLE_EVERY_MAX)
    return Lmax * 0.5 + SAFETY_PAD_LEN

def build_big_cell_and_crops():
    """根据 NX, NY（目标裁剪窗的标称大小）与反射安全净距，生成大域与多个裁剪窗。"""
    nx0, ny0 = NX, NY
    scale = random.uniform(0.8, 1.2)
    nx_win = max(32, int(round(nx0 * scale)))
    ny_win = max(32, int(round(ny0 * scale)))

    Mmin_units = required_min_margin_units(RESOLUTION)
    Mmin_px = int(math.ceil(Mmin_units * RESOLUTION))

    ring_need_px = 1 + PATCH_RADIUS + KEEP_MARGIN

    nx_big = nx_win + 2 * (Mmin_px + DPML_PX + ring_need_px)
    ny_big = ny_win + 2 * (Mmin_px + DPML_PX + ring_need_px)

    cell = mp.Vector3(nx_big / RESOLUTION, ny_big / RESOLUTION, 0)
    dpml = DPML_PX / RESOLUTION

    half_wx = 0.5 * nx_win / RESOLUTION
    half_wy = 0.5 * ny_win / RESOLUTION
    crop_phys = (-half_wx, half_wx, -half_wy, half_wy)

    n_crops = random.randint(NCROPS_MIN, NCROPS_MAX)
    crops = []
    for _ in range(n_crops):
        s = random.uniform(0.85, 1.15)
        nxw = max(32, int(round(nx_win * s)))
        nyw = max(32, int(round(ny_win * s)))
        half_wx_i = 0.5 * nxw / RESOLUTION
        half_wy_i = 0.5 * nyw / RESOLUTION

        hx_big = 0.5 * nx_big / RESOLUTION
        hy_big = 0.5 * ny_big / RESOLUTION
        pad_units = (Mmin_px + DPML_PX + ring_need_px) / RESOLUTION
        max_dx = max(0.0, hx_big - pad_units - half_wx_i)
        max_dy = max(0.0, hy_big - pad_units - half_wy_i)
        cx = random.uniform(-max_dx*0.8, max_dx*0.8)
        cy = random.uniform(-max_dy*0.8, max_dy*0.8)
        crops.append((cx, cy, nxw, nyw))
    return cell, dpml, nx_big, ny_big, crops

def rect_pix_from_center_size(cx, cy, nxw, nyw, nx_big, ny_big):
    """把以 (cx,cy) 为中心、nxw×nyw 像素的窗，转为像素坐标矩形 (x0,x1,y0,y1)（含端点）。
       注意：Meep 取 array 时得到的是 (nx,ny)，我们随后会转为 (ny,nx) 再做索引。"""
    dx_px = int(round(cx * RESOLUTION))
    dy_px = int(round(cy * RESOLUTION))
    x_center = nx_big // 2 + dx_px
    y_center = ny_big // 2 + dy_px
    x0 = x_center - nxw // 2
    x1 = x0 + nxw - 1
    y0 = y_center - nyw // 2
    y1 = y0 + nyw - 1
    return (x0, x1, y0, y1)

def rotate_rect_yx(rect_yx, shape_yx, k):
    """对 (ny,nx) 形状下的矩形 (x0,x1,y0,y1) 做 90°*k 旋转后的像素坐标。
       这里 rect_yx 的 x/y 是按 (ny,nx) 的列/行坐标（x=列，y=行）。"""
    x0, x1, y0, y1 = rect_yx
    ny, nx = shape_yx
    k = k % 4
    if k == 0:
        return (x0, x1, y0, y1), (ny, nx)
    if k == 1:
        new_y0 = nx - 1 - x1
        new_y1 = nx - 1 - x0
        new_x0 = y0
        new_x1 = y1
        return (new_x0, new_x1, new_y0, new_y1), (nx, ny)
    if k == 2:
        new_x0 = nx - 1 - x1
        new_x1 = nx - 1 - x0
        new_y0 = ny - 1 - y1
        new_y1 = ny - 1 - y0
        return (new_x0, new_x1, new_y0, new_y1), (ny, nx)

    new_y0 = x0
    new_y1 = x1
    new_x0 = ny - 1 - y1
    new_x1 = ny - 1 - y0
    return (new_x0, new_x1, new_y0, new_y1), (nx, ny)


def extract_edge_samples_from_window(ex_yx, ey_yx, hz_yx, rect_yx_rot, keep_margin=KEEP_MARGIN):
    """
    输入是已旋转到 NORTH 朝向后的 yx 场与对应的裁剪窗像素矩形。
    结构与旧版 extract_edge_samples 一致；目标 ghost 取自窗外一行（真实场）。
    """
    x0, x1, y0, y1 = rect_yx_rot
    ny, nx = hz_yx.shape
    dx = 1.0 / RESOLUTION
    r = PATCH_RADIUS

    y_in = y1
    y_out = y1 + 1
    if y_out >= ny or y_in - 1 < 0:
        return np.zeros((0, 38), np.float32), np.zeros((0, 1), np.float32)

    xs, ys1 = [], []
    x_start = x0 + keep_margin + r
    x_end_inclusive = x1 - keep_margin - r
    if x_end_inclusive < x_start:
        return np.zeros((0, 38), np.float32), np.zeros((0, 1), np.float32)

    for x in range(x_start, x_end_inclusive + 1):
        cols = [x + k for k in range(-r, r+1)]
        rows = [y_in, y_in - 1]

        hz_patch = np.array([hz_yx[rows[j], cols] for j in range(2)]).reshape(-1)
        et_patch = np.array([ex_yx[rows[j], cols] for j in range(2)]).reshape(-1)
        en_patch = np.array([ey_yx[rows[j], cols] for j in range(2)]).reshape(-1)

        feats_deriv = _phys_derivs_from_patch(ex_yx, ey_yx, hz_yx, y_in, y_in - 1, x, dx)
        feats_pos   = _pos_features_along_edge(x, x_start, x_end_inclusive, x1 - x0 + 1)

        feat = np.concatenate([hz_patch, et_patch, en_patch, feats_deriv, feats_pos], axis=0).astype(np.float32)
        ghost = hz_yx[y_out, x] 
        xs.append(feat)
        ys1.append([ghost])

    X = np.stack(xs, axis=0) if xs else np.zeros((0, 38), np.float32)
    Y1 = np.stack(ys1, axis=0).astype(np.float32) if ys1 else np.zeros((0, 1), np.float32)
    return X, Y1

def extract_corner_samples_from_window(ex_yx, ey_yx, hz_yx, rect_yx_rot):
    """
    NORTH 朝向下取右上角（x_in = x1, y_in = y1）邻近 patch，返回 1×(feat) 与 1×3 的 (gN,gE,gNE)。
    为了提高角点数据量，外层会做小倍数复制（CORNER_REP）。
    """
    x0, x1, y0, y1 = rect_yx_rot
    ny, nx = hz_yx.shape
    dx = 1.0 / RESOLUTION
    r = PATCH_RADIUS

    y_in = y1
    x_in = x1
    x = x_in - 1
    y = y_in - 1

    if (x - r < x0) or (x + r >= nx) or (y_in + 1 >= ny) or (x_in + 1 >= nx) or (y < 0):
        return np.zeros((0, 38), np.float32), np.zeros((0, 3), np.float32)

    cols = [x + k for k in range(-r, r+1)]
    rows = [y_in, y_in - 1]

    hz_patch = np.array([hz_yx[rows[j], cols] for j in range(2)]).reshape(-1)
    et_patch = np.array([ex_yx[rows[j], cols] for j in range(2)]).reshape(-1)
    en_patch = np.array([ey_yx[rows[j], cols] for j in range(2)]).reshape(-1)

    feats_deriv = _phys_derivs_from_patch(ex_yx, ey_yx, hz_yx, y_in, y_in - 1, x, dx)

    x_start = x0 + KEEP_MARGIN + r
    x_end   = x1 - KEEP_MARGIN - r
    feats_pos = _pos_features_along_edge(x, x_start, x_end, x1 - x0 + 1)

    feat = np.concatenate([hz_patch, et_patch, en_patch, feats_deriv, feats_pos], axis=0).astype(np.float32)

    gN  = hz_yx[y_in + 1, x]
    gE  = hz_yx[y, x_in + 1]
    gNE = hz_yx[y_in + 1, x_in + 1]
    Y3 = np.array([gN, gE, gNE], dtype=np.float32).reshape(1, 3)
    X1 = feat.reshape(1, -1)
    if CORNER_REP > 1:
        X1 = np.repeat(X1, CORNER_REP, axis=0)
        Y3 = np.repeat(Y3, CORNER_REP, axis=0)
    return X1, Y3

def assemble_sequences_for_side(frames_yx, rect_yx, k_rot):
    """
    给定一组帧（每帧是 (ex_yx,ey_yx,hz_yx)），将该窗旋转到 k_rot 后统一为 NORTH，
    在所有时间步上收集边与角的序列样本，形状与旧版保持一致。
    """
    X_edge_seq, Y1_seq, X_corner_seq, Y3_seq = [], [], [], []
    ny0, nx0 = frames_yx[0][0].shape
    rect_rot, shape_rot = rotate_rect_yx(rect_yx, (ny0, nx0), k_rot)

    for ex_yx, ey_yx, hz_yx in frames_yx:
        exN, eyN, hzN = rot_fields_tmz(ex_yx, ey_yx, hz_yx, k_rot)
        Xe_t, Y1_t = extract_edge_samples_from_window(exN, eyN, hzN, rect_rot, keep_margin=KEEP_MARGIN)
        Xc_t, Y3_t = extract_corner_samples_from_window(exN, eyN, hzN, rect_rot)
        if Xe_t.shape[0] > 0:
            X_edge_seq.append(Xe_t)
            Y1_seq.append(Y1_t)
        if Xc_t.shape[0] > 0:
            X_corner_seq.append(Xc_t)
            Y3_seq.append(Y3_t)

    def stack_seq(seq_list):
        if not seq_list:
            return np.zeros((0, 0, 0), np.float32)
        return np.stack(seq_list, axis=1).astype(np.float32)

    return stack_seq(X_edge_seq), stack_seq(Y1_seq), stack_seq(X_corner_seq), stack_seq(Y3_seq)


def build_one_episode(ep_id: int, scene_tag: str = "random"):
    """
    一次大域仿真 + 随机多裁剪窗；返回所有窗/四边的序列样本。

    scene_tag:
        "random"  : 原始随机几何 + 随机源（保持你之前的分布）
        "internal": 模拟 comparison 脚本中 internal_source 场景的几何 + 内部点源，
                    针对 CPML 边界原本更占优的那类分布做专门强化。
    """
    cell, dpml, nx_big, ny_big, crops = build_big_cell_and_crops()

    cx0, cy0, nxw0, nyw0 = crops[0]
    rect0_pix = rect_pix_from_center_size(cx0, cy0, nxw0, nyw0, nx_big, ny_big)

    half_wx = 0.5 * nxw0 / RESOLUTION
    half_wy = 0.5 * nyw0 / RESOLUTION
    rect0_phys = (-half_wx, half_wx, -half_wy, half_wy)

    if scene_tag == "internal":
        geometry, sources = make_internal_source_like_scene(rect0_phys, RESOLUTION)
    else:
        geometry = random_geometry_in_rect(rect0_phys, RESOLUTION)
        sources = random_sources_in_rect(rect0_phys, RESOLUTION)

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=[mp.PML(dpml)],
        resolution=RESOLUTION,
        dimensions=2,
        sources=sources,
    )
    sim.init_sim()

    for _ in range(T_WARMUP):
        sim.fields.step()

    sample_every = random.randint(SAMPLE_EVERY_MIN, SAMPLE_EVERY_MAX)

    frames_all = []
    for _ in range(T_STEPS):
        for _ in range(sample_every):
            sim.fields.step()
        ex = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)
        ey = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
        hz = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)
        ex_yx = _as_yx(ex, ex.shape[0])
        ey_yx = _as_yx(ey, ey.shape[0])
        hz_yx = _as_yx(hz, hz.shape[0])
        frames_all.append((ex_yx, ey_yx, hz_yx))

    all_X_edge, all_Y1, all_X_corner, all_Y3 = [], [], [], []
    all_ep_edge, all_ep_corner = [], []
    for (cx, cy, nxw, nyw) in crops:
        rect_pix = rect_pix_from_center_size(cx, cy, nxw, nyw, nx_big, ny_big)
        rect_yx = (rect_pix[0], rect_pix[1], rect_pix[2], rect_pix[3])

        for k_rot in (0, 1, 2, 3):
            Xe, Y1, Xc, Y3 = assemble_sequences_for_side(frames_all, rect_yx, k_rot)

            if Xe.size > 0:
                if Xe.shape[0] > MAX_SAMPLES_PER_EP:
                    idx = np.random.choice(Xe.shape[0], size=MAX_SAMPLES_PER_EP, replace=False)
                    Xe, Y1 = Xe[idx], Y1[idx]
                all_X_edge.append(Xe)
                all_Y1.append(Y1)
                all_ep_edge.append(np.full((Xe.shape[0],), ep_id, dtype=np.int32))

            if Xc.size > 0:
                all_X_corner.append(Xc)
                all_Y3.append(Y3)
                all_ep_corner.append(np.full((Xc.shape[0],), ep_id, dtype=np.int32))

    return all_X_edge, all_Y1, all_X_corner, all_Y3, all_ep_edge, all_ep_corner


# ============== 主流程 ==============

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_X_edge, all_Y1, all_X_corner, all_Y3 = [], [], [], []
    all_ep_edge, all_ep_corner = [], []

    n_internal = int(round(N_EPISODES * INTERNAL_SCENE_FRACTION))

    for ep_id in tqdm(range(N_EPISODES), desc="采集(大域+多窗, 含 internal_source-like 场景)"):
        scene_tag = "internal" if ep_id < n_internal else "random"

        Xe_list, Y1_list, Xc_list, Y3_list, ep_e_list, ep_c_list = build_one_episode(ep_id, scene_tag=scene_tag)

        all_X_edge += Xe_list
        all_Y1     += Y1_list
        all_X_corner += Xc_list
        all_Y3       += Y3_list
        all_ep_edge  += ep_e_list
        all_ep_corner += ep_c_list


    X_edge = np.concatenate(all_X_edge, axis=0) if all_X_edge else np.zeros((0, T_STEPS, 38), np.float32)
    Y1     = np.concatenate(all_Y1, axis=0)     if all_Y1     else np.zeros((0, T_STEPS, 1), np.float32)
    ep_edge = np.concatenate(all_ep_edge, axis=0) if all_ep_edge else np.zeros((0,), np.int32)

    X_corner = np.concatenate(all_X_corner, axis=0) if all_X_corner else np.zeros((0, T_STEPS, 38), np.float32)
    Y3       = np.concatenate(all_Y3, axis=0)       if all_Y3       else np.zeros((0, T_STEPS, 3), np.float32)
    ep_corner = np.concatenate(all_ep_corner, axis=0) if all_ep_corner else np.zeros((0,), np.int32)

    X = np.concatenate([X_edge, X_corner], axis=0)
    Y = np.zeros((X.shape[0], T_STEPS, 3), np.float32)
    mask_edge = np.zeros((X.shape[0],), bool)
    mask_corner = np.zeros((X.shape[0],), bool)
    ep_ids = np.zeros((X.shape[0],), np.int32)

    n_edge = X_edge.shape[0]
    if n_edge > 0:
        Y[:n_edge, :, 0:1] = Y1
        mask_edge[:n_edge] = True
        ep_ids[:n_edge] = ep_edge
    if X_corner.shape[0] > 0:
        X[n_edge:n_edge+X_corner.shape[0]] = X_corner
        Y[n_edge:n_edge+X_corner.shape[0]] = Y3
        mask_corner[n_edge:n_edge+X_corner.shape[0]] = True
        ep_ids[n_edge:n_edge+X_corner.shape[0]] = ep_corner

    X = np.clip(X, -CLIP_VALUE, CLIP_VALUE)
    Y = np.clip(Y, -CLIP_VALUE, CLIP_VALUE)

    np.savez_compressed(DATA_DIR / f"{DATASET_NAME}.npz",
                        X=X, Y=Y,
                        mask_edge=mask_edge, mask_corner=mask_corner,
                        ep_ids=ep_ids.astype(np.int32), T=T_STEPS)

    with open(DATA_DIR / f"{DATASET_NAME}.meta.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            note=(
                "True-BC via big-domain + interior crops; "
                "ghost=one-ring outside crop from true fields; "
                "multi-crops/episode; four sides per crop; "
                "2x5 patch + deriv + pos; CFL-safe to avoid PML echoes; "
                "geometry/sources confined to crop; "
                "corner oversampled."
            ),
            T=T_STEPS, warmup=T_WARMUP, patch_radius=PATCH_RADIUS,
            crops_per_episode=f"{NCROPS_MIN}-{NCROPS_MAX}",
            corner_rep=CORNER_REP
        ), f, ensure_ascii=False, indent=2)

    print("✅ dataset saved successfully.")

if __name__ == "__main__":
    main()