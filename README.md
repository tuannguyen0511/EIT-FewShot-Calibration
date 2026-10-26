# Few-shot calibration of simulation-trained EIT reconstruction networks

Code accompanying *When Is Few-Shot Calibration Enough? A Pre-Specified Multi-System Evaluation of Simulation-Trained EIT Reconstruction Networks* (submitted to IEEE Transactions on Instrumentation and Measurement).

## What is included
| Path | Content |
|---|---|
| `eit_calib/model.py` | FiLM-conditioned U-Net with GroupNorm (8,593,890 parameters for 208 voltages; 5,888 GroupNorm affine parameters). |
| `eit_calib/preprocess.py` | Voltage standardization, J-map back-projection and network inputs. |
| `eit_calib/calibrate.py` | Few-shot normalization-affine calibration (Adam, lr 1e-2, 100 full-batch steps, BCE on support logits inside the tank) and prediction. |
| `eit_calib/metrics.py` | Dice, IoU, false-positive pixels, area ratio, HD95; case-bootstrap CI and exact sign-flip test. |
| `examples/demo_calibration.py` | Data-free example of the API. |
| `simulation/simulate_template.m` | EIDORS template for generating simulated training pairs (a guide; set geometry, inclusion and noise models for your instrument). |
| `prespecification/` | System 1 configuration lock and System 3 pre-specification, amendment and lock file (SHA-256). |
| `results/` | Per-case results of the two confirmatory tests (System 1: 20 cases; System 3: 21 targets). |

Trained weights (`backbone_A_16el.pt`, 16 electrodes/208 voltages; `backbone_A32_KTC.pt`, 32 electrodes/2356 voltages) are attached to the archived release ([Zenodo DOI]).

```python
from eit_calib.model import E2EMultiViewShapeNet
from eit_calib.calibrate import calibrate, predict
model = E2EMultiViewShapeNet(voltage_dim=208)
model.load_state_dict(torch.load("backbone_A_16el.pt", map_location="cpu"))
calibrated = calibrate(model, images, voltages, masks, rois)   # k labelled phantoms of your system
sigma, support = predict(calibrated, test_images, test_voltages)
```

## Data
System 2 (KIT4, Hauptmann et al., arXiv:1704.01178) and System 3 (KTC 2023, Zenodo doi:10.5281/zenodo.10986692) are public. System 1 data are available from their original authors (Deng et al., IEEE Trans. Ind. Electron., 2026). Simulated training data, the full experiment pipeline, additional backbones and further result tables are available from the corresponding authors on reasonable request.

## Citation
Please cite the paper if you use this code ([citation to be added]).

## License
MIT (see `LICENSE`).
