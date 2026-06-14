# Submission Manifest

This file records the active submission artifact and recent leaderboard probes.
Submission quota rules are defined in `docs/7_submission_quota_strategy.md`.

## Active Champion

| Field | Value |
| :--- | :--- |
| File | `submission.csv` |
| Source | `scratch/pull_disagreement_analysis/submission_alpha_0.50.csv` |
| Kaggle notebook | `tuannm3812/stellar-classification-disagreement-analysis-v1` |
| Public score | `0.97069` |
| Status | Active champion |

The root `submission.csv` is frozen to the static 5-external blend with
`alpha=0.50`.

## Rejected Probe

| File | Changed Rows | Public Score | Decision |
| :--- | ---: | :--- | :--- |
| `scratch/targeted_disagreement_candidates/close3v2_gs_local_margin_0.00.csv` | 84 | `0.97025` | Reject |

The rejected probe flipped only `GALAXY`/`STAR` rows where the five core
external submissions split 3-2 and the local stack selected the minority side.
The score drop indicates that local tie-breaking is not a useful routing signal
for this disagreement subset.
