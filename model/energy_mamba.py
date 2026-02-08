import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from base.model import BaseModel


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class GCN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, H, A_hat):
        Hp = H.permute(0, 2, 1, 3)
        out = torch.matmul(A_hat, self.linear(Hp))
        return F.gelu(out).permute(0, 2, 1, 3)


class GEMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.gcn = GCN(d_model)
        self.norm = RMSNorm(d_model)
        self.proj_fwd = nn.Linear(2 * d_model, d_model)
        self.proj_bwd = nn.Linear(2 * d_model, d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A_hat):
        B, N, T, D = H.shape
        H_norm = self.norm(H)
        Z = self.gcn(H_norm, A_hat)
        h_flat = H_norm.reshape(B * N, T, D)
        z_flat = Z.reshape(B * N, T, D)
        hz = torch.cat([h_flat, z_flat], dim=-1)
        fwd_out = self.mamba_fwd(self.proj_fwd(hz))
        bwd_out = self.mamba_bwd(self.proj_bwd(hz.flip(1))).flip(1)
        out = (fwd_out + bwd_out).reshape(B, N, T, D)
        return H + self.dropout(out)


class GEMambaStage(nn.Module):
    def __init__(self, d_model, d_state, num_layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            GEMambaBlock(d_model, d_state, dropout) for _ in range(num_layers)
        ])

    def forward(self, H, A_hat):
        for blk in self.blocks:
            H = blk(H, A_hat)
        return H


class EnergyMamba(BaseModel):
    def __init__(self, d_model, d_state, num_layers, depth, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.depth = max(1, depth)

        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, self.seq_len, d_model) * 0.02)

        self.encoder_stages = nn.ModuleList([
            GEMambaStage(d_model, d_state, num_layers, dropout)
            for _ in range(self.depth)
        ])

        down_blocks = max(0, self.depth - 1)
        self.downsamples = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            for _ in range(down_blocks)
        ])
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
            for _ in range(down_blocks)
        ])
        self.decoder_stages = nn.ModuleList([
            GEMambaStage(d_model, d_state, num_layers, dropout)
            for _ in range(down_blocks)
        ])
        self.skip_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(down_blocks)
        ])

        self.bottleneck = GEMambaStage(d_model, d_state, num_layers, dropout)

        self.out_norm = RMSNorm(d_model)
        self.head_lo = nn.Linear(d_model, self.horizon)
        self.head_mid = nn.Linear(d_model, self.horizon)
        self.head_up = nn.Linear(d_model, self.horizon)

    def _apply_conv(self, x, layer):
        B, N, T, D = x.shape
        x = x.reshape(B * N, T, D).permute(0, 2, 1)
        x = layer(x).permute(0, 2, 1)
        return x.reshape(B, N, -1, D)

    def _match_time(self, x, target):
        T = x.shape[2]
        if T == target:
            return x
        if T > target:
            return x[:, :, :target, :]
        return F.pad(x, (0, 0, 0, target - T))

    def forward(self, x, A_hat):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3)
        H = self.input_proj(x) + self.pos_emb

        skips = []
        for idx, stage in enumerate(self.encoder_stages):
            H = stage(H, A_hat)
            if idx < len(self.downsamples):
                skips.append(H)
                H = self._apply_conv(H, self.downsamples[idx])

        H = self.bottleneck(H, A_hat)

        for stage, up, proj in zip(
            reversed(self.decoder_stages),
            reversed(self.upsamples),
            reversed(self.skip_projs),
        ):
            H = self._apply_conv(H, up)
            skip = skips.pop()
            H = self._match_time(H, skip.shape[2])
            H = torch.cat([H, skip], dim=-1)
            H = proj(H)
            H = stage(H, A_hat)

        H = self._match_time(H, self.seq_len)
        H_out = self.out_norm(H[:, :, -1, :])

        lo = self.head_lo(H_out).permute(0, 2, 1).unsqueeze(-1)
        mid = self.head_mid(H_out).permute(0, 2, 1).unsqueeze(-1)
        up = self.head_up(H_out).permute(0, 2, 1).unsqueeze(-1)

        return lo, mid, up
