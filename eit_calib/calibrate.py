"""Few-shot normalization-affine calibration of a frozen EIT reconstruction network.

Only the affine parameters (gamma, beta) of every GroupNorm layer are trained; all other
weights stay frozen. Settings follow the paper's locked protocol: Adam, lr 1e-2,
100 full-batch steps, pixel-wise BCE on the support logits inside the region of interest.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_affine_parameters(model: nn.Module):
    """Return the GroupNorm weight and bias tensors of the model."""
    params = []
    for m in model.modules():
        if isinstance(m, nn.GroupNorm) and m.affine:
            params += [m.weight, m.bias]
    return params


def calibrate(model: nn.Module, images, voltages, masks, rois, steps: int = 100, lr: float = 1e-2,
              view_id: int = 0, seed: int = 0):
    """Calibrate a copy of `model` on k labelled phantoms and return it.

    images:   float tensor (k, 2, 64, 64)  -- network image input [z0, J-map]
    voltages: float tensor (k, V)          -- standardized voltages
    masks:    float tensor (k, 1, 64, 64)  -- reference support masks (0/1)
    rois:     bool  tensor (k, 1, 64, 64)  -- region of interest (pixels inside the tank)
    """
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    params = norm_affine_parameters(m)
    for p in params:
        p.requires_grad_(True)
    m.eval()  # dropout off; GroupNorm uses per-sample statistics
    opt = torch.optim.Adam(params, lr=lr)
    view = torch.full((images.shape[0],), view_id, dtype=torch.long, device=images.device)
    n_pix = rois.sum()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = m(images, voltages, view)["mask_logit"]
        loss = F.binary_cross_entropy_with_logits(logits[rois], masks[rois], reduction="sum") / n_pix
        loss.backward()
        opt.step()
    for p in params:
        p.requires_grad_(False)
    return m


@torch.no_grad()
def predict(model: nn.Module, images, voltages, view_id: int = 0):
    """Return conductivity estimate and binary support mask (sigmoid(logit) > 0.5)."""
    model.eval()
    view = torch.full((images.shape[0],), view_id, dtype=torch.long, device=images.device)
    out = model(images, voltages, view)
    return out["pred_sigma"], (torch.sigmoid(out["mask_logit"]) > 0.5)
