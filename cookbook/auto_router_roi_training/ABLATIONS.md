# Card and calibration ablations

These comparisons hold the boundary at the original default: capability base 0.5 with step 0.1, or V2 quality gap 0.05. All coefficients were fitted on the training split. This table reports every raw card and the fixed middle regularization strength of 10 for calibrated variants; the JSON contains all strengths. No variant is chosen using these held-out results

| Pair | Classifier | Card | Adjustment | Solved | Cost | Savings | Efficient Brier |
|---|---|---|---|---|---:|---:|---:|
| sonnet_opus | v2 | original | none | 23/25 | $8.619 | 0.9% | 0.080 |
| sonnet_opus | v2 | original | per_model | 23/25 | $8.716 | -0.2% | 0.279 |
| sonnet_opus | v2 | original | task_conditioned | 23/25 | $8.716 | -0.2% | 0.296 |
| sonnet_opus | cap | original | none | 24/25 | $7.439 | 14.5% | 0.074 |
| sonnet_opus | cap | original | per_model | 23/25 | $8.716 | -0.2% | 0.288 |
| sonnet_opus | cap | original | task_conditioned | 23/25 | $8.716 | -0.2% | 0.288 |
| sonnet_opus | v2 | research | none | 23/25 | $8.380 | 3.7% | 0.076 |
| sonnet_opus | v2 | research | per_model | 23/25 | $8.716 | -0.2% | 0.288 |
| sonnet_opus | v2 | research | task_conditioned | 23/25 | $8.716 | -0.2% | 0.297 |
| sonnet_opus | cap | research | none | 24/25 | $7.433 | 14.6% | 0.066 |
| sonnet_opus | cap | research | per_model | 23/25 | $8.710 | -0.1% | 0.288 |
| sonnet_opus | cap | research | task_conditioned | 23/25 | $8.710 | -0.1% | 0.303 |
| sonnet_opus | v2 | trained_card | none | 23/25 | $8.711 | -0.1% | 0.169 |
| sonnet_opus | v2 | trained_card | per_model | 23/25 | $8.711 | -0.1% | 0.288 |
| sonnet_opus | v2 | trained_card | task_conditioned | 23/25 | $8.711 | -0.1% | 0.304 |
| sonnet_opus | cap | trained_card | none | 24/25 | $7.705 | 11.4% | 0.125 |
| sonnet_opus | cap | trained_card | per_model | 23/25 | $8.710 | -0.1% | 0.288 |
| sonnet_opus | cap | trained_card | task_conditioned | 23/25 | $8.710 | -0.1% | 0.299 |
| luna_sol | v2 | original | none | 25/25 | $5.887 | 23.5% | 0.114 |
| luna_sol | v2 | original | per_model | 25/25 | $7.713 | -0.2% | 0.293 |
| luna_sol | v2 | original | task_conditioned | 25/25 | $7.713 | -0.2% | 0.309 |
| luna_sol | cap | original | none | 23/25 | $0.454 | 94.1% | 0.095 |
| luna_sol | cap | original | per_model | 25/25 | $7.548 | 1.9% | 0.265 |
| luna_sol | cap | original | task_conditioned | 25/25 | $7.504 | 2.5% | 0.276 |
| luna_sol | v2 | research | none | 25/25 | $7.132 | 7.3% | 0.114 |
| luna_sol | v2 | research | per_model | 25/25 | $7.714 | -0.2% | 0.268 |
| luna_sol | v2 | research | task_conditioned | 25/25 | $7.714 | -0.2% | 0.274 |
| luna_sol | cap | research | none | 23/25 | $0.454 | 94.1% | 0.080 |
| luna_sol | cap | research | per_model | 25/25 | $7.707 | -0.1% | 0.293 |
| luna_sol | cap | research | task_conditioned | 25/25 | $7.707 | -0.1% | 0.315 |
| luna_sol | v2 | trained_card | none | 25/25 | $7.708 | -0.1% | 0.228 |
| luna_sol | v2 | trained_card | per_model | 25/25 | $7.708 | -0.1% | 0.272 |
| luna_sol | v2 | trained_card | task_conditioned | 25/25 | $7.708 | -0.1% | 0.250 |
| luna_sol | cap | trained_card | none | 23/25 | $1.936 | 74.9% | 0.131 |
| luna_sol | cap | trained_card | per_model | 25/25 | $7.707 | -0.1% | 0.293 |
| luna_sol | cap | trained_card | task_conditioned | 25/25 | $7.707 | -0.1% | 0.304 |

Lower Brier means more accurate probabilities on these realized attempts. A lower Brier score can still yield worse routing at a fixed boundary. Compare these fixed-boundary controls with the separately frozen validation-selected profiles in REPORT.md. These exploratory comparisons reuse the same 25 tasks, so selecting a new winner here would require another holdout
