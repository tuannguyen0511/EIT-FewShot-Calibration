# K2 PREREGISTRATION V1 — few-shot system calibration on the independent 32-electrode system (KTC 2023)
Author: brain (Claude), 2026-10-24. Written BEFORE any prediction on the 21 evaluation targets exists. The SHA256 of this file is recorded in BRAIN_STATE and must appear in the K2 lock file. No amendment after the lock.

## 1. Fixed inputs (assert SHA256 before use)
- Backbone KTC-B: `04_RUNS\K1B_KTCB\best_clean_dice_head.pt`, SHA `5bff866ba378ef79995783b8f4cbddc7234e7f902adb127f55c6839fabd07d1f` (synthetic TEST Dice_head 0.9130, accepted).
- Operators: A_KTC `05_RESULTS\RUNBOOK\K1A_KTC_SYNTH_AP\A_KTC_sparse_2356x4096.npy` (SHA 83742e97…a3e1); M (SHA 3f19ca96…baf0); z0 operator (SHA 488e5c73…da68).
- Data: `00_CONTEXT\SPECS\KTC_PATH_MANIFEST_V1.json` (25 targets, SHA per file); reference `ref.mat`. GT: 64×64 support masks `05_RESULTS\RUNBOOK\K0_KTC_AUDIT\K0R2_<id>_mask64_R2.npy` and 3-class `K0R2_<id>_classes64_R2.npy` (area-fraction rule, organiser frame).
- Input construction exactly as K1b (Amendment A4 + §2-AT): ΔU = Uel − Uelref (full 2356 data); z0 = 4×4 block of interpolateRecoToPixGrid(M·ΔU), outside disk 0, same image-centre mode as K1b training; J-map = reshape(A_KTCᵀ vnorm(ΔU)); view_id = 0.

## 2. Sets
- Calibration set (fixed, official training targets): TR1, TR2, TR3, TR4 (k = 4; TR3 conductive-only, TR4 resistive-only, TR1/TR2 both).
- Test set (single use): the 21 evaluation targets EV_L1_1 … EV_L7_3 (full data). Unit of analysis = target, n = 21.
- ROI = pixel centres inside the tank radius (organiser frame); all metrics inside ROI.

## 3. Methods (all on the same frozen KTC-B; settings copied from the C3 lock, no tuning on KTC)
| Method | Trainable | Optimiser | lr | Steps |
|---|---|---|---|---|
| Frozen | none | – | – | 0 |
| **CAL-GN (primary)** | all GroupNorm γ, β | Adam, full batch (k = 4), model.eval() | 1e-2 | 100 |
| CAL-LoRA | LoRA as in C3 | Adam | 1e-3 | 100 |
| CAL-FULL | all parameters | Adam | 1e-4 | 100 |
| CAL-BitFit | all bias parameters | Adam | 1e-2 (CMP DEV-selected) | 100 |
| Classical (context) | organiser one-step linear reconstruction + Otsu2 (organiser defaults) | – | – | – |
Loss for all calibrated methods: pixel BCE on mask_logit vs 64×64 support mask, mean over ROI pixels of the 4 calibration targets (identical to C0/C3). Calibrated parameters are restored after each method (no carry-over).

## 4. Endpoints and decision rules
**Primary (confirmatory):** mean over the 21 targets of ΔDice_head = Dice(CAL-GN) − Dice(frozen). Two-sided exact sign-flip test on the 21 paired differences (all 2²¹ sign patterns), α = 0.05; 95 % case-bootstrap CI (10,000 resamples, seed 20261024).
- **REPLICATED** if mean ΔDice > 0, p < 0.05 and CI lower bound > 0.
- **NOT REPLICATED** otherwise (reported as such; no re-analysis to rescue it).
**Secondary (Holm over this family of 4):** (S1) P_harm = fraction of targets with ΔDice < −0.005 (descriptive, reported with Wilson CI); (S2) CAL-GN − CAL-FULL, (S3) CAL-GN − CAL-LoRA, (S4) CAL-GN − CAL-BitFit: paired mean, exact sign-flip p, bootstrap CI.
**Descriptive:** IoU, FP pixels, area ratio, HD95; frozen and calibrated Dice per target; classical Otsu2 Dice (support = class > 0) and official organiser score (KTCScoring.scoringFunction) for classical, frozen and CAL-GN (3-class from network: support m > 0.5, class 2 if σ̂ > 1, class 1 if σ̂ ≤ 1, upsampled to 256×256 by nearest neighbour).

## 5. Pre-specified quantitative prediction (from REAL02 DEV, H1R-4)
Model fitted on REAL02 DEV (before KTC): ΔDice_i = 0.627 − 0.666 · Dice_frozen,i (material term ignored, coefficient not significant). For every KTC target compute the predicted ΔDice from its frozen Dice. Report: (P1) predicted vs observed mean ΔDice; (P2) Spearman ρ between predicted and observed per-target ΔDice with bootstrap CI. Prediction CONFIRMED if ρ > 0 with CI lower bound > 0. Label: exploratory, pre-specified. (Expected caveat: ceiling/regression-to-the-mean.)

## 6. Exploratory (labelled as such)
- k-curve: CAL-GN with every subset of TR1–TR4 of size 1, 2, 3 (4 + 6 + 4 = 14 runs), evaluated on the 21 targets.
- Subgroups: targets containing a conductive object vs resistive-only; level 1–7 breakdown (difficulty levels as target identifiers only).
- Cross-system transfer is not tested (different electrode count).

## 7. Integrity
One script; frozen and all calibrated predictions on the 21 targets are produced once, after the lock file (`K2_LOCK.json`: SHAs of this preregistration, script, checkpoint, operators, manifest, GT masks, settings) is written. No inspection of evaluation predictions before the lock; no re-run after outputs exist (fail-closed if the output folder exists). Never write synthetic contrast ranges.
