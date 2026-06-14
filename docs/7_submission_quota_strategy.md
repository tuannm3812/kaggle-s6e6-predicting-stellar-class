# Submission Quota Strategy

Kaggle notebook runs and leaderboard submissions should be treated as separate
steps.

## Notebook Runs

Use notebook pushes/runs for:

- Rebuilding `submission.csv` from a reproducible notebook.
- Pulling diagnostics, logs, and candidate files.
- Comparing candidates by row-level diffs before any leaderboard submission.

Notebook execution is the preferred way to reproduce artifacts, but it does not
replace leaderboard scoring. A notebook output still needs to be submitted to
the competition to receive a public score.

## Leaderboard Submissions

Only submit a CSV when all of these are true:

1. The candidate changes a clearly defined subset of rows.
2. The candidate has a written rule and a generated diff file.
3. The expected direction is not already rejected by prior submissions.
4. The root `submission.csv` champion remains available as fallback.

Do not submit:

- Blind alpha sweeps.
- Variants that only differ by a few arbitrary rows.
- Router rules that repeat the rejected local tie-breaker direction.
- Multiple candidates from the same hypothesis unless the first result supports
  that hypothesis.

## Current Policy

The active champion is the static 5-external blend with `alpha=0.50`
(`0.97069`). The rejected probe
`close3v2_gs_local_margin_0.00.csv` scored `0.97025`, so local tie-breaking on
close `GALAXY`/`STAR` external splits is blocked as a submission direction.

## Account Usage

Use the account with confirmed mounted-data access for notebook generation. For
this project, `tuannm3812` successfully ran
`stellar-classification-disagreement-analysis-v1` with:

- `flexonafft/stellar-data`
- `playground-series-s6e6`
- `tuannm3812/stellar-classification-baseline-modeling`

Use the secondary account only after confirming it can mount the same external
dataset. A failed mount can silently degrade the artifact and waste a
submission.

## Practical Workflow

1. Run the diagnostic notebook and pull outputs.
2. Generate candidate CSVs and `_diff.csv` files locally.
3. Review `targeted_candidate_summary.csv`.
4. Submit at most one candidate per hypothesis.
5. Record every public score in `docs/6_submission_manifest.md`.
6. If a candidate loses materially, mark the hypothesis rejected and stop that
   branch.
