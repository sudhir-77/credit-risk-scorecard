# Resume notes

## Summary

This project compares a weight-of-evidence logistic scorecard with a monotonic LightGBM default model on the UCI Default of Credit Card Clients dataset (30,000 accounts, Taiwan 2005). The champion is chosen on a validation set, then discrimination, calibration, adverse-action reason codes, fairness, and population stability are measured once on a held-out test set.

## Skills and tools

Python, pandas, scikit-learn, LightGBM, SHAP, matplotlib, pytest, logistic regression, gradient boosting, weight of evidence, calibration, fairness metrics

## Draft bullets

- Built a credit-default model on 30,000 real card accounts with monotonic LightGBM at 0.781 AUROC and 0.423 KS vs. 0.761 for a WoE logistic scorecard, calibrated to 0.012 expected calibration error
- Generated top-4 adverse-action reason codes from SHAP for every declined test applicant and checked adverse impact ratio across sex and age bands (0.919 minimum)
- Documented stability (PSI) and a champion/challenger recommendation in a model-risk write-up

## GitHub description

Monotonic LightGBM vs a WoE credit scorecard with calibration, SHAP reason codes, and fair-lending checks on UCI default data.
