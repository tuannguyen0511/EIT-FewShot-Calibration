"""Minimal, data-free demonstration of the calibration API (random tensors only).

Replace the random tensors by real calibration phantoms (inputs built with
eit_calib.preprocess.network_inputs) and load released weights to reproduce the method.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eit_calib.model import E2EMultiViewShapeNet
from eit_calib.calibrate import calibrate, predict, norm_affine_parameters

model = E2EMultiViewShapeNet(voltage_dim=208)
# model.load_state_dict(torch.load("weights/backbone_A_16el.pt", map_location="cpu"))
print("norm-affine parameters:", sum(p.numel() for p in norm_affine_parameters(model)))

k = 5
images = torch.randn(k, 2, 64, 64)
voltages = torch.randn(k, 208)
masks = (torch.rand(k, 1, 64, 64) > 0.9).float()
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
rois = ((xx ** 2 + yy ** 2) <= 1).expand(k, 1, 64, 64)

calibrated = calibrate(model, images, voltages, masks, rois, steps=5)
sigma, support = predict(calibrated, images, voltages)
print("output shapes:", tuple(sigma.shape), tuple(support.shape))
