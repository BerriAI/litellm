# Training and selection record

The learning target is model complementarity: route cheaply when Luna can do the job, and use Sol when its extra cost is likely to produce an additional solve. Better marginal probability accuracy alone is not the selection objective

## Paired examples

| Split | Benchmark | Tasks | Both fail | Sol only | Luna only | Both solve |
|---|---|---:|---:|---:|---:|---:|
| train | lcb | 50 | 14 | 9 | 3 | 24 |
| train | mbpp | 80 | 7 | 2 | 3 | 68 |
| train | swe | 25 | 5 | 1 | 0 | 19 |
| train | terminal | 5 | 0 | 1 | 0 | 4 |
| validation | lcb | 20 | 2 | 6 | 2 | 10 |
| validation | mbpp | 29 | 4 | 2 | 2 | 21 |
| validation | swe | 10 | 1 | 1 | 1 | 7 |
| validation | terminal | 5 | 2 | 0 | 1 | 2 |

There are 13 Sol-only examples in fitting and 9 in validation. Only two fitting examples are agentic rescues, one SWE-bench and one Terminal-Bench. Terminal validation has no Sol-only example. That limits evidence about rescue detection on unseen terminal tasks; zero validation losses there cannot establish rescue recall

## Variations

One training-only synthesis produces the empirical card supplement. It uses measured outcomes and fallible classifier/reviewer descriptions, without task identities, solutions or validation-driven rewrites. Original cards are a separate control

The deterministic head families are raw threshold tuning, scalar probability calibration, task-dependent per-model calibration, four-way paired outcomes, Sol-only rescue probability and paired benefit per predicted incremental dollar. Regularization and fixed threshold grids are selected using validation only. Upfront inputs come from one classifier response. Post-attempt heads additionally use Luna output, a fallible Luna review and task-supplied public checks where available

The strong-success-retention objective minimizes validation cost while losing no Sol-only successes and preserving Sol aggregate quality in each benchmark. The quality-first-under-sol-budget objective maximizes validation solves under Sol total cost, permitting task swaps. Always-Sol is an explicit fallback. None of these objective names guarantees performance on unseen tasks

## Frozen selected profiles

| Classifier | Stage | Objective | Cards | Family | Threshold | Validation solved | Validation cost | Sol-only losses |
|---|---|---|---|---|---:|---:|---:|---:|
| cap | upfront | strong_success_retention | original | benefit_per_dollar | 1.0 | 50/64 | $5.8544 | 0 |
| cap | upfront | quality_first_under_sol_budget | original | scalar_calibration | 0.1 | 52/64 | $5.4692 | 2 |
| cap | review | strong_success_retention | original | benefit_per_dollar | 0.25 | 49/64 | $5.6530 | 0 |
| cap | review | quality_first_under_sol_budget | empirical | paired | 0.075 | 53/64 | $3.7201 | 2 |
| cap | either | strong_success_retention | original | benefit_per_dollar | 0.25 | 49/64 | $5.6530 | 0 |
| cap | either | quality_first_under_sol_budget | empirical | paired | 0.075 | 53/64 | $3.7201 | 2 |
| v2 | upfront | strong_success_retention | original | benefit_per_dollar | 1.0 | 51/64 | $5.4640 | 0 |
| v2 | upfront | quality_first_under_sol_budget | original | scalar_calibration | 0.075 | 53/64 | $5.4524 | 2 |
| v2 | review | strong_success_retention | original | per_model | -0.02 | 51/64 | $6.7043 | 0 |
| v2 | review | quality_first_under_sol_budget | empirical | per_model | 0.15 | 53/64 | $3.3074 | 2 |
| v2 | either | strong_success_retention | original | benefit_per_dollar | 1.0 | 51/64 | $5.4640 | 0 |
| v2 | either | quality_first_under_sol_budget | empirical | per_model | 0.15 | 53/64 | $3.3074 | 2 |

These are development results. Compare held-out results in REPORT.md before interpreting them as an improvement. Both selected upfront classifier profiles keep the original card and use fitted local decisions. Empirical cards remain a measured variation and are selected by some review cascades

There are 1,892 policy candidates, 118 serialized models, 12 selected profiles and 40 prespecified family ablations. An independent refit reproduced every coefficient, candidate statistic and selection exactly. The selection hash and source/data hashes are recorded in selection_frozen.json; reproduction evidence is in training_reproduction.json

The threshold unit depends on the target. Marginal and paired gaps are probability differences, rescue is a Sol-only probability, and benefit_per_dollar is expected additional success per predicted dollar. A threshold of 0.05 is not interchangeable across these families
