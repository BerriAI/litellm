# Auto Router training experiment

Profiles were fitted on 83 DeepSWE tasks and selected on 30 repository-disjoint validation tasks before inspecting live grades. The live comparison uses 25 native-image-eligible SWE-bench Verified tasks and four current solver models at high effort. Each solver runs once per task; frozen task-pinned policies reuse those attempts and add their judge cost

The baselines use the original classifier prompts and boundaries under the same judge settings: capability base 0.5 with step 0.1, and V2 gap 0.05 with neutral model-name profiles. This isolates classifier changes. It does not reproduce a production router that reclassifies every turn or escalates during an attempt

| Pair | Policy | Solved | Cost | Savings vs capable | Efficient tasks | Lost / gained |
|---|---|---:|---:|---:|---:|---:|
| Sonnet / Opus | Sonnet only | 24/25 | $7.422 | 14.7% | 25 | 1 / 2 |
| Sonnet / Opus | Opus only | 23/25 | $8.699 | 0.0% | 0 | 0 / 0 |
| Sonnet / Opus | Original capability | 24/25 | $7.439 | 14.5% | 25 | 1 / 2 |
| Sonnet / Opus | Original V2 | 23/25 | $8.619 | 0.9% | 1 | 0 / 0 |
| Sonnet / Opus | Trained capability, 0 pp validation allowance | 24/25 | $7.705 | 11.4% | 22 | 1 / 2 |
| Sonnet / Opus | Trained capability, 2 pp validation allowance | 24/25 | $7.633 | 12.3% | 22 | 1 / 2 |
| Sonnet / Opus | Trained capability, 5 pp validation allowance | 24/25 | $7.633 | 12.3% | 22 | 1 / 2 |
| Sonnet / Opus | Trained V2, 0 pp validation allowance | 23/25 | $8.711 | -0.1% | 0 | 0 / 0 |
| Sonnet / Opus | Trained V2, 2 pp validation allowance | 23/25 | $8.711 | -0.1% | 0 | 0 / 0 |
| Sonnet / Opus | Trained V2, 5 pp validation allowance | 23/25 | $8.711 | -0.1% | 0 | 0 / 0 |
| Luna / Sol | Luna only | 23/25 | $0.444 | 94.2% | 25 | 2 / 0 |
| Luna / Sol | Sol only | 25/25 | $7.697 | 0.0% | 0 | 0 / 0 |
| Luna / Sol | Original capability | 23/25 | $0.454 | 94.1% | 25 | 2 / 0 |
| Luna / Sol | Original V2 | 25/25 | $5.887 | 23.5% | 7 | 0 / 0 |
| Luna / Sol | Trained capability, 0 pp validation allowance | 23/25 | $1.234 | 84.0% | 23 | 2 / 0 |
| Luna / Sol | Trained capability, 2 pp validation allowance | 24/25 | $5.643 | 26.7% | 8 | 1 / 0 |
| Luna / Sol | Trained capability, 5 pp validation allowance | 23/25 | $1.234 | 84.0% | 23 | 2 / 0 |
| Luna / Sol | Trained V2, 0 pp validation allowance | 23/25 | $2.086 | 72.9% | 21 | 2 / 0 |
| Luna / Sol | Trained V2, 2 pp validation allowance | 23/25 | $0.891 | 88.4% | 23 | 2 / 0 |
| Luna / Sol | Trained V2, 5 pp validation allowance | 25/25 | $7.713 | -0.2% | 0 | 0 / 0 |

The fitted settings below were selected on validation. The allowance is a constraint on average net validation loss, not the V2 gap threshold and not a production guarantee

| Pair | Classifier | Validation allowance | Card | Probability adjustment | Boundary |
|---|---|---:|---|---|---|
| sonnet_opus | cap | 0 pp | trained_card | none | base_threshold=0.5, threshold_step=0 |
| sonnet_opus | cap | 2 pp | research | none | base_threshold=0.72, threshold_step=0 |
| sonnet_opus | cap | 5 pp | research | none | base_threshold=0.72, threshold_step=0 |
| sonnet_opus | v2 | 0 pp | trained_card | task_conditioned (regularization 1) | max_quality_gap=0.1649 |
| sonnet_opus | v2 | 2 pp | trained_card | task_conditioned (regularization 1) | max_quality_gap=0.1649 |
| sonnet_opus | v2 | 5 pp | trained_card | task_conditioned (regularization 1) | max_quality_gap=0.1649 |
| luna_sol | cap | 0 pp | original | task_conditioned (regularization 100) | base_threshold=0.4565, threshold_step=0 |
| luna_sol | cap | 2 pp | original | task_conditioned (regularization 1) | base_threshold=0.5, threshold_step=0 |
| luna_sol | cap | 5 pp | original | task_conditioned (regularization 10) | base_threshold=0.4544, threshold_step=0 |
| luna_sol | v2 | 0 pp | original | none | max_quality_gap=0.08 |
| luna_sol | v2 | 2 pp | research | none | max_quality_gap=0.09 |
| luna_sol | v2 | 5 pp | original | per_model (regularization 1) | max_quality_gap=0.2561 |

Equal solve counts can hide different successful tasks. Twenty-five tasks cannot establish small quality differences; the JSON includes paired repository-cluster intervals. These intervals are exploratory with few repositories. If every observed paired difference is zero, the empirical bootstrap interval is also zero and cannot estimate unseen failures. Even zero lost successes in 25 independent trials permits an 11.3% one-sided 95% binomial upper bound on that event rate; repository dependence weakens that inference. Public-data training uses different budgets and serving configurations, so this is a transfer test

The initial x86 runs were excluded because of environment activation and emulator startup failures. Native tasks were selected by the same seeded repository/task ordering, skipping unavailable ARM images before any live grade inspection. The sample is not representative of every SWE-bench platform or repository

The OpenAI adapter pilot was excluded because the gateway split sequential output blocks across choices and the stock harness discarded its tool calls. A later accounting correction restarted all three in-flight Anthropic attempts to capture the billed cost of malformed responses. Four completed Anthropic attempts were retained after confirming complete accounting. Grading container-name conflicts were retried with model-specific containers, preserving the solver attempts. One incomplete Sol attempt was archived and retried after gateway rate limits exhausted transport retries before submission; its $0.345 cost is excluded from the policy comparison and recorded separately. The final Sol grade exceeded 900 seconds during host slowdown; an unmodified-image CLI control also took 102 seconds. Only the saved patch was regraded with a 3600-second infrastructure deadline. These amendments, excluded attempts and unchanged frozen-selection hash are recorded in protocol.json

Costs include billed solver responses, including malformed replies, plus the applicable classifier forecast. Host compute, image downloads and excluded infrastructure pilots are separate experiment expenses. The replay does not demonstrate mid-task escalation, inherited-state rescue, or a new repeated stochastic router run

The JSON also reports held-out Brier scores for per-model probabilities and mean squared error for the V2 predicted gap. Those diagnostics measure probability accuracy separately from the routing threshold. One attempt per model per task does not reveal a task's true solve probability
