# model.py
import math
import torch
import torch.nn as nn
from config import D_INPUT, D_MODEL, D_STATE, N_LAYERS

def kaiming_init_(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SimpleSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int, p_drop: float = 0.1):
        super().__init__()
        assert d_state % 2 == 0 and d_state > 0
        self.d_model = d_model
        self.d_state = d_state
        self.half = d_state // 2

        self.in_proj  = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.D = nn.Linear(d_model, d_model)

        self.a_raw = nn.Parameter(torch.randn(self.half) * 0.1)  # radius
        self.w_raw = nn.Parameter(torch.randn(self.half) * 0.1)  # angle

        self.gate = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p_drop)

        self.apply(kaiming_init_)

    def forward(self, x, state):
        Bx = self.in_proj(x)
        s_re = state[:, :self.half]
        s_im = state[:, self.half:]

        radius = torch.sigmoid(self.a_raw) * 0.999
        theta  = math.pi * torch.tanh(self.w_raw)
        c, s = torch.cos(theta), torch.sin(theta)

        n_re = radius * (s_re * c - s_im * s)
        n_im = radius * (s_re * s + s_im * c)
        next_state = torch.cat([n_re, n_im], dim=-1) + Bx

        core = self.out_proj(next_state)
        y = core * torch.sigmoid(self.gate(x))
        y = self.drop(y)
        y = y + self.D(x)
        return y, next_state

class NP_SSM_CRBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_ln = nn.LayerNorm(D_INPUT, elementwise_affine=False)

        layers = [nn.Linear(D_INPUT, D_MODEL), nn.SiLU(), nn.Dropout(0.05)]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(D_MODEL, D_MODEL), nn.SiLU(), nn.Dropout(0.05)]
        self.encoder = nn.Sequential(*layers)

        self.ssm = SimpleSSM(D_MODEL, D_STATE, p_drop=0.1)
        self.norm = nn.LayerNorm(D_MODEL)

        self.head1 = nn.Linear(D_MODEL, 1)
        self.head3 = nn.Linear(D_MODEL, 3)

        self.apply(kaiming_init_)

    @torch.no_grad()
    def init_state(self, batch_size: int, device=None):
        device = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch_size, D_STATE, device=device)

    def forward(self, x, state):
        x_enc = self.encoder(self.in_ln(x))
        y_ssm, next_state = self.ssm(x_enc, state)
        y = self.norm(x_enc + y_ssm)
        out1 = self.head1(y)
        out3 = self.head3(y)
        return out1, out3, next_state
