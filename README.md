# Predicting Electric Vehicle Purchases

**Kaggle Playground Series — Season 6, Episode 9** | Binary classification (AUC) on synthetic data

> Final result: **400th of 3,576 teams (top 11.2%)** — private AUC **0.94543**, best public AUC **0.94646**.

## Overview

The dataset was synthetically generated (SDG) from a real underlying dataset, which made
**feature engineering the core of the competition**: the strongest signal was not in the
columns themselves but in the fingerprints the data generator left behind.

| Stage | Score (AUC) |
|---|---|
| First baseline | 0.9417 |
| Final public best | **0.94646** |
| Final private (official) | **0.94543** |
| Own-pipeline-only blend (public) | 0.94620 |

Fully self-built pipeline (no external submissions blended): LightGBM 6-seed rank-average
+ XGBoost + CatBoost greedy blend → OOF 0.94628 / public 0.94620.

## Key techniques

- **Generator-leak feature engineering** — the single biggest win:
  digit decomposition of numeric columns, modular features (`x % 7/9/11/13/97`),
  quant ladders (`x // 50/100/.../5000`), value-frequency encoding, smooth keys
  (`str`, `//100`, `//1000`), and SDG artifact flags (e.g. income == 30000 spike).
- **Triple target encoding** — 10 keys × 3 smoothing levels, refit inside each fold
  (no leakage).
- **Multi-GBM ensemble** — LightGBM, XGBoost and CatBoost trained on the identical
  recipe; predictions blended in rank space.
- **Seed averaging** — 3 → 6 seeds gave a reliable +0.00005 by averaging away variance.
- **OOF-stacked greedy blending** — an out-of-fold blending framework over 33 candidate
  members, with weights chosen by greedy search on OOF AUC.
- **Strict validation discipline** — every decision was made on 10-fold stratified OOF,
  which is what moved the final standing from 549 (public) to 400 (private).

## Lessons learned

1. **Visible bias ≠ exploitable signal.** Residual analysis found segments where the
   model erred systematically — but corrections gained ~0, because the model had
   already priced that signal in. Always test a "fix" before trusting the diagnosis.
2. **Blends have a resolution limit.** Beyond ~0.9464, more mixing stopped helping;
   gains came from a better single model, not more members.
3. **Seed/variance control is free accuracy.** Averaging across seeds and fold splits
   cancels random tittering while systematic signal survives.
4. **OOF reliability pays off at the end.** A 0.0005 OOF–public gap meant the private
   shake-out *improved* the ranking instead of hurting it.

## Repository structure

Scripts follow the actual experimentation timeline (v2 → v10):

| File | Purpose |
|---|---|
| `eda.py` | Exploratory data analysis |
| `model_v2.py` … `model_v9.py` | Iterating feature families (interactions → digit leak) |
| `model_v10.py` | Final LightGBM recipe (leak + smooth keys + triple TE), resumable, seed-parametric |
| `model_v10xgb.py`, `model_v10cat.py` | Same recipe on XGBoost / CatBoost |
| `v10_seedavg.py` | Multi-seed rank averaging |
| `stack_blend.py` | OOF-stacked greedy blending framework |
| `own_stack_final.py` | Final own-pipeline blend (LGBM 6-seed + XGB + CatBoost) |
| `analyze_residuals.py`, `segment_shift_search.py` | Residual/segment analysis + correction ablation |
| `optuna_v10.py`, `optuna_v7.py`, `optuna_xgb.py` | Hyperparameter search experiments |
| `build_final_rules.py` | Final submission construction |
| `match_original.py`, `digit_match.py` | SDG ↔ original-dataset matching probes |

Final submission CSVs are included as evidence; intermediate experiment submissions are
excluded for brevity.

## Notes

- Competition data is **not** included — download it from the
  [competition page](https://www.kaggle.com/competitions/playground-series-s6e9).
- Third-party OOF submissions used during stacking are deliberately **not** published here.
- All training scripts support fold-level resume: re-running the same command after an
  interruption continues from completed folds.
