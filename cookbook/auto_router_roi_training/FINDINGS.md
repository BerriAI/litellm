> **Historical benchmark invalidated:** the 25-task live run exposed upstream fixes through Git objects outside the task checkout. Its quality and savings figures are withdrawn. These files retain the audit record, not evidence of routing improvements. Fresh isolated experiments are in progress

# What the training experiment showed

Both implementations now support runnable, opt-in trained profiles. Training covered three cards, raw probabilities, per-model calibration, task-dependent calibration, and pair-specific boundaries, including combinations. All fitting and policy selection used the DeepSWE training and validation splits. The 25 fresh SWE-bench tasks were used only for evaluation

The selected profiles did not establish a consistent improvement over the original routers. Most paid more for the same solve count or traded away solves for savings. They remain experimental configurations, with the original defaults preserved

## Comparison with the existing routers

For Sonnet 5 / Opus 5, Sonnet-only solved 24/25 for $7.422 and Opus-only solved 23/25 for $8.699. Original capability matched Sonnet's 24 solves at $7.439. The primary trained capability profile also solved 24, but cost $7.705. Original V2 solved 23 for $8.619; trained V2 selected Opus for every task, solved 23, and cost $8.711

For Luna / Sol, Luna-only solved 23/25 for $0.444 and Sol-only solved 25/25 for $7.697. Original V2 solved the same 25 tasks for $5.887, saving 23.5% versus Sol. The primary trained V2 profile kept the original card and raw probabilities but raised the gap boundary from 0.05 to 0.08. It cost $2.086 and solved 23, losing both tasks Luna failed. Its lower cost therefore came with an observed quality loss. The primary trained capability profile also solved 23 and cost $1.234, versus original capability's 23 for $0.454

The other validation allowances are reported in BENCHMARK.md in the PRs and REPORT.md in the full bundle. The Luna/Sol capability profile selected with a two-point validation allowance solved 24 for $5.643. That is a quality/cost tradeoff, not a profile that maximizes both. The most permissive selected V2 calibration chose Sol everywhere and added judge cost

## Cards, probability calibration, and boundaries had different effects

The research rewrite improved the capability classifier's efficient-model Brier score from 0.0736 to 0.0662 for Sonnet, and from 0.0953 to 0.0804 for Luna. With the original boundary, it still routed all 25 tasks to the efficient solver. This is a descriptive signal for better card wording, without an observed solve-rate improvement or a meaningful solver-cost advantage. It was not promoted to a new winner using the held-out outcomes

The training-derived priors and learned probability adjustments transferred poorly. The primary trained V2 profile predicted a mean 37-point Opus advantage, while the realized paired difference was a 4-point Sonnet advantage. Its Sonnet probability averaged 0.434 against observed success of 0.96. The primary Luna capability calibration averaged 0.463 against observed success of 0.92. These estimates and Brier scores show a calibration problem on this workload, independently of the threshold

Threshold tuning cannot repair that probability error. The Luna/Sol 0.05-to-0.08 comparison also shows that a boundary can be too permissive even when the mean predicted gap is close to the observed average gap. Request-level ranking and the placement of harmful downgrades matter, as well as average calibration

## What to train next

Use paired outcomes collected with the same agent, model effort, tools and budget as the intended deployment, with a new repository-held-out evaluation set. Treat these 25 tasks as used evaluation data. DeepSWE supplied useful fitting data, but this experiment does not establish whether its domain, task difficulty, budget or serving differences caused the failed transfer

Train a model-pair estimate of the expected quality difference and the risk that only the stronger solver succeeds, alongside per-model expected total attempt cost. Some Sonnet attempts needed 71-84 model calls, so nominal token prices alone do not capture the routing opportunity. Keep task labels and model-specific probabilities as inputs, and test shrinking learned corrections toward the original probabilities when matching evidence is sparse

Keep the research-only card rewrite as a candidate for that next evaluation. Do not increase the boundary merely to make these held-out results look better. The unchanged Luna/Sol V2 configuration is a useful quality-preserving control on this sample, and the efficient-only model is an essential savings control

## Scope of the evidence

These results come from 100 fresh solver attempts, one attempt per model per task, plus 300 fresh classifier forecasts. The policies are frozen, task-pinned paired replays over those attempts, with measured judge cost added. They do not measure per-turn reclassification, escalation or additional independently sampled router runs. Twenty-five tasks cannot establish a small production quality-loss guarantee
