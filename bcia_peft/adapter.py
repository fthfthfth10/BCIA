from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class LightweightDSMEncoder(nn.Module):
    """MsRE's 2→16→32→64 stem, without the original projection."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 4, stride=4, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 2, stride=2, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 2, stride=2, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )

    def forward(self, dsm, valid):
        valid = valid & torch.isfinite(dsm)
        masked = torch.where(valid, dsm, 0.0)
        mask = valid.to(dsm.dtype)
        coverage = F.avg_pool2d(mask, kernel_size=16, stride=16)
        features = self.encoder(torch.cat((masked, mask), dim=1))
        return torch.where(coverage > 0, features, 0.0), coverage


class BCIAOutput(NamedTuple):
    residual: torch.Tensor
    coverage: torch.Tensor
    z_rgb: torch.Tensor
    z_dsm: torch.Tensor
    pred_dsm: torch.Tensor
    pred_rgb: torch.Tensor
    u_rgb: torch.Tensor
    u_dsm: torch.Tensor
    w_rgb: torch.Tensor
    w_dsm: torch.Tensor
    dsm_contribution: torch.Tensor
    mixed: torch.Tensor


class BCIA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_projection = nn.Linear(1024, 64)
        self.g_rgb_to_dsm = nn.Conv2d(64, 64, 3, padding=1)
        self.g_dsm_to_rgb = nn.Conv2d(64, 64, 3, padding=1)
        self.router = nn.Conv2d(257, 2, 1)
        self.mix = nn.Conv2d(128, 64, 1)
        self.output_projection = nn.Linear(64, 1024)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, spatial, z_dsm, coverage):
        batch = spatial.shape[0]
        z_rgb = self.rgb_projection(spatial).transpose(1, 2).reshape(batch, 64, 32, 32)
        supported = coverage > 0
        z_dsm = torch.where(supported, z_dsm, 0.0)
        pred_dsm = self.g_rgb_to_dsm(z_rgb)
        pred_rgb = torch.where(supported, self.g_dsm_to_rgb(z_dsm), 0.0)
        u_dsm = torch.where(supported, z_dsm - pred_dsm, 0.0)
        u_rgb = z_rgb - pred_rgb
        routed = self.router(torch.cat((z_rgb, z_dsm, u_rgb, u_dsm, coverage), dim=1))
        w_rgb, w_dsm = routed.sigmoid().split(1, dim=1)
        w_dsm = torch.where(supported, w_dsm, 0.0)
        dsm_contribution = torch.where(supported, w_dsm * u_dsm, 0.0)
        mixed = self.mix(torch.cat((w_rgb * u_rgb, dsm_contribution), dim=1))
        residual = self.output_projection(mixed.flatten(2).transpose(1, 2))
        return BCIAOutput(residual, coverage, z_rgb, z_dsm, pred_dsm, pred_rgb,
                          u_rgb, u_dsm, w_rgb, w_dsm, dsm_contribution, mixed)


def prediction_losses(output):
    """Separate coverage-weighted MSEs; no segmentation loss coupling."""
    weight = output.coverage
    denom = (weight.sum() * 64).clamp_min(1.0)
    dsm_error = (output.pred_dsm - output.z_dsm.detach()).square()
    rgb_error = (output.pred_rgb - output.z_rgb.detach()).square()
    return {
        "rgb_to_dsm": (dsm_error * weight).sum() / denom,
        "dsm_to_rgb": (rgb_error * weight).sum() / denom,
    }
