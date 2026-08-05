# Bank Marketing — Loan/Subscription Classifier

Predict who subscribes (`y`) from the UCI Bank Marketing data.

## Files
- `data_check.py` / `eda.py` — dataset overview (shape, missing, target balance, correlations).
- `eda_bivariate.py` — conversion rate + `n` per category.
- `eda_plots.py` — charts (target balance, conversion by category, numeric-by-target, heatmap).
- `train_model.py` — Logistic Regression baseline + evaluation plots.

## Key choices
- Drop `duration` (leakage). `pdays==999` → `was_contacted_before` flag.
- OneHot + scaling inside a `Pipeline`; `stratify` split; `class_weight="balanced"` for the 11% imbalance.
- Metrics: ROC-AUC / PR-AUC (not accuracy).

## Result (baseline)
ROC-AUC ≈ 0.80 · PR-AUC ≈ 0.46 (vs 0.11 baseline) · recall ≈ 0.65.

## Run
```bash
python train_model.py
```
