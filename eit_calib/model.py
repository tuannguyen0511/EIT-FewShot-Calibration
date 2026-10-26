"""FiLM-conditioned U-Net with GroupNorm used in the paper (architecture unchanged from the trained models)."""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BG_VALUE = 1.0

BASE_CH = 32

COND_DIM = 192

V_HIDDEN_DIM = 320

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)

        if cond_dim is not None:
            self.film = nn.Linear(cond_dim, out_ch * 2)
        else:
            self.film = None

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.gn2(x)

        if self.film is not None and cond is not None:
            gb = self.film(cond)
            gamma, beta = gb.chunk(2, dim=1)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            x = x * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta

        x = F.silu(x)
        return x

class E2EMultiViewShapeNet(nn.Module):
    def __init__(
        self,
        image_ch: int = 2,
        voltage_dim: int = 208,
        base_ch: int = BASE_CH,
        cond_dim: int = COND_DIM,
        v_hidden_dim: int = V_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.v_encoder = nn.Sequential(
            nn.Linear(voltage_dim, v_hidden_dim),
            nn.LayerNorm(v_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(v_hidden_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
        )

        self.view_emb = nn.Embedding(5, cond_dim)

        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        c = base_ch

        self.enc1 = ConvBlock(image_ch, c, cond_dim)
        self.enc2 = ConvBlock(c, c * 2, cond_dim)
        self.enc3 = ConvBlock(c * 2, c * 4, cond_dim)
        self.enc4 = ConvBlock(c * 4, c * 8, cond_dim)

        self.bottleneck = ConvBlock(c * 8, c * 16, cond_dim)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8, cond_dim)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4, cond_dim)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, cond_dim)

        self.sigma_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, 1, 1),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, 1, 1),
        )

    def forward(
        self,
        image_input: torch.Tensor,
        voltage: torch.Tensor,
        view_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vc = self.v_encoder(voltage)
        ve = self.view_emb(view_id)

        cond = self.cond_fuse(torch.cat([vc, ve], dim=1))

        e1 = self.enc1(image_input, cond)
        e2 = self.enc2(F.max_pool2d(e1, 2), cond)
        e3 = self.enc3(F.max_pool2d(e2, 2), cond)
        e4 = self.enc4(F.max_pool2d(e3, 2), cond)

        b = self.bottleneck(F.max_pool2d(e4, 2), cond)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1), cond)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), cond)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond)

        delta = torch.tanh(self.sigma_head(d1))
        pred_sigma = BG_VALUE + delta

        mask_logit = self.mask_head(d1)

        return {
            "pred_sigma": pred_sigma,
            "mask_logit": mask_logit,
        }
