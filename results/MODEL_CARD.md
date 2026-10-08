# Model card: credit default scorecard

This is a portfolio study on a public dataset. It is not a production underwriting model, it was not used to decision real applicants, and it has no regulatory approval.

## Purpose

Compare a weight-of-evidence logistic scorecard with a monotonic LightGBM model on the probability that a credit-card client defaults next month. The write-up records discrimination, calibration, adverse-action reason codes, fair-lending ratios, and population stability.

## Data

- Source: Default of Credit Card Clients, UCI Machine Learning Repository, I-Cheng Yeh and Che-hui Lien.
- License: CC BY 4.0.
- Population: 30,000 Taiwanese credit-card clients from 2005.
- Target: `label` = 1 if the client defaulted the next month.
- Split: stratified 60/20/20 train/validation/test, seed 42. Binning, LightGBM hyperparameters, isotonic calibration, and the decline cutoff use train and validation only. The test set is scored once.
- Protected attributes: sex and age are excluded from model inputs and used only in the fairness report.

## Models

- Challenger: weight-of-evidence logistic scorecard. Base score 600, 20 points to double the odds of non-default. Bins are fit on the training set. `C` and class weight are chosen by validation AUROC.
- Candidate: LightGBM with monotone constraints. Repayment status, utilization, and average/max delinquency cannot decrease predicted risk as they rise. Credit limit and payment amounts cannot increase predicted risk as they rise. Education, marriage, and raw bill amounts are unconstrained because their direction is not reliable.
- Both raw scores are passed through isotonic regression fit on validation predictions only.
- Champion selection uses validation AUROC of the calibrated scores. Selected champion: `monotonic_lightgbm`.
- Decline rule: calibrated probability of default >= 0.243243. The cutoff maximizes Youden's J (TPR minus FPR) on the validation set.

## Test results

| Metric | Champion | WoE scorecard |
|---|---:|---:|
| AUROC | 0.7808 | 0.7612 |
| KS | 0.4234 | 0.3980 |
| Gini | 0.5615 | 0.5223 |
| Brier | 0.1353 | 0.1392 |
| ECE (10 equal-width bins) | 0.0123 | 0.0113 |

Validation AUROC used for selection: monotonic LightGBM 0.7867, scorecard 0.7730.

## Reason codes

For every test applicant whose calibrated probability is at or above the cutoff, the report lists four reason codes. Codes are the monotonic features with the largest TreeSHAP contributions to the default log-odds (LightGBM `pred_contrib`), restricted to features whose value is on the adverse side of a training reference. `PAY_0_HIGH` means a high recent delinquency code, which the constraint treats as higher risk. `PAY_AMT1_LOW` means a low payment, which the constraint treats as higher risk. Declined test applicants: 1693.

## Fair lending

Adverse impact ratio is the group's approval rate divided by the approval rate of the largest group on that attribute. Equal-opportunity difference is the gap in approval rate among accounts that did not default (group minus reference). Sex reference: `female`. Age-band reference: `30-39`.

Minimum AIR: 0.9191. Maximum AIR: 1.0000.

- No group has an adverse impact ratio outside 0.80-1.25 at this cutoff.

These ratios describe this test sample. They are not a legal determination of disparate impact.

## Population stability

PSI compares the training distribution with the test distribution. Edges are the training quantiles. On a random split, PSI should be small. The largest value in this run is BILL_AMT1 at 0.0024.

| Feature | PSI | Bins |
|---|---:|---:|
| LIMIT_BAL | 0.0015 | 10 |
| EDUCATION | 0.0003 | 4 |
| MARRIAGE | 0.0005 | 3 |
| PAY_0 | 0.0013 | 5 |
| PAY_2 | 0.0001 | 4 |
| PAY_3 | 0.0001 | 4 |
| PAY_4 | 0.0001 | 4 |
| PAY_5 | 0.0002 | 3 |
| PAY_6 | 0.0002 | 4 |
| BILL_AMT1 | 0.0024 | 10 |
| BILL_AMT2 | 0.0010 | 10 |
| BILL_AMT3 | 0.0010 | 10 |
| BILL_AMT4 | 0.0023 | 10 |
| BILL_AMT5 | 0.0006 | 10 |
| BILL_AMT6 | 0.0007 | 10 |
| PAY_AMT1 | 0.0019 | 9 |
| PAY_AMT2 | 0.0011 | 9 |
| PAY_AMT3 | 0.0012 | 8 |
| PAY_AMT4 | 0.0009 | 8 |
| PAY_AMT5 | 0.0017 | 8 |
| PAY_AMT6 | 0.0007 | 8 |
| UTIL_1 | 0.0021 | 10 |
| AVG_PAY_STATUS | 0.0010 | 8 |
| MAX_PAY_STATUS | 0.0014 | 5 |
| AVG_PAY_AMT | 0.0017 | 10 |

A common reading is PSI below 0.10 as stable, 0.10 to 0.25 as a moderate shift, and above 0.25 as a large shift. This dataset has no application date, so the check is a split diagnostic rather than a through-the-time monitor.

## Recommendation

Use `monotonic_lightgbm` as the champion and the weight-of-evidence scorecard as the challenger on this dataset, based on validation AUROC. Keep the scorecard as the interpretable benchmark: its points table is the audit trail for the linear model. Do not deploy either model for credit decisions. Any real use would need a time-based validation sample, a documented adverse-action policy, and a review of the fairness flags above.

## Limitations

- The data are Taiwanese credit cards from 2005, not a current US portfolio.
- There is no reject inference and no application or behavioral timeline.
- Monotone constraints are modeling choices. They can hide a real non-monotone pattern.
- Sex and age are not inputs, but other features can proxy for them. The fairness table is the check, not a fix.
- Isotonic calibration and early stopping both see the validation labels, so validation metrics are optimistic. Test metrics are the ones to quote.
- Reason codes explain this model. They are not a compliance sign-off.
