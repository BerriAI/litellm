# Capability router

The capability router asks an LLM to forecast each configured candidate's probability of completing the current task, then selects the cheapest candidate whose forecast clears the configured reliability threshold. This is separate from the complexity router and its local or trained heuristic classifiers

## Train capability cards

Capability-card training uses outcomes from the real task harness. It does not replace the LLM classifier. The classifier still reads the task and cards at runtime; the trained artifact improves the evidence and policy around its forecast by:

- assigning each card rule a boundary from observed end-to-end outcomes and a 95% Wilson interval;
- attaching a smoothed observed success probability to each matched rule;
- fitting monotonic probability calibration for each candidate;
- tuning the global probability threshold and boundary step on validation tasks; and
- reporting raw and calibrated Brier score, log loss, calibration error, solve rate, cost, and quality-cost utility on untouched test tasks

Create one JSONL row for every `(task, candidate, run)`:

```json
{"benchmark":"swe-bench-verified","task_id":"django__django-12345","task_family":"django/django","split":"train","model":"efficient","primary_rule":"R3","raw_p_solve":0.72,"success":1.0,"estimated_cost":0.18}
```

`raw_p_solve` and `primary_rule` come from the capability classifier's routing decision. `success` must come from the benchmark's end-to-end verifier, not an LLM estimate. Repeated runs may use the same task and model; the trainer averages them when measuring routing quality

Every benchmark task must have an explicit `train`, `validation`, or `test` split and outcomes for every configured candidate. A task cannot cross splits. Set `task_family` to a repository, domain, task generator, or another shared origin when related tasks could leak. The trainer rejects a family found across splits. For a general preset, assign entire benchmark families to a split and deduplicate related tasks before training. A random row split over near-duplicate tasks overstates generalization

Run:

```shell
python -m litellm.router_strategy.capability_router.training outcomes.jsonl \
  --config capability-router.json \
  --artifact-output trained-capability-router.json \
  --quality-weight 0.7
```

The artifact wraps a ready-to-use `CapabilityRouterConfig`. Its candidate rules carry learned boundaries and `probability_calibration` bins. The report printed to stdout compares the trained route with the original untrained cards on the same test tasks

### Seed-card design

Write rules around observable failure mechanisms, not broad labels such as "easy", "hard", "coding", or "reasoning". A useful rule tells the classifier what property decides success: whether the procedure is explicit, the required state is inspectable, a validator covers the output, policy conditions conflict, an action is irreversible, or complete search has no boundary. Match the hardest requirement rather than an easy side task

Keep rule text shared across candidates when possible, then let each candidate's learned boundary express its coverage. Do not put prices, routing instructions, thresholds, or claimed percentages in a card. Those belong to deterministic policy and the learned calibration artifact

Retain an existing boundary when a rule has no training observations. With observations, the trainer changes it to supported only when the lower confidence bound clears the reliability target, unsupported only when the upper bound misses it, and uncertain otherwise

## Benchmark protocol

Use the same candidate model revisions, agent, tools, task budget, provider settings, and number of attempts for every arm. At minimum compare:

1. always efficient;
2. always capable;
3. the original Switchyard-style qualitative cards and raw `p_solve`;
4. the trained cards and calibrated `p_solve`; and
5. an oracle computed from the recorded candidate outcomes

Report the complete solve-rate versus cost curve rather than one threshold. Tune cards, calibration, and thresholds on training and validation tasks only. Run the final configuration once on the test split and keep that result unchanged

Recommended executable public suites are Terminal-Bench 2.1, SWE-bench Verified or Pro, tau2-bench, AppWorld, BFCL, and ToolSandbox. RouterBench is useful as a cheap classifier and calibration smoke test, but it is not evidence of agentic end-to-end performance

### SWE-bench Verified coding run

The coding run used all 500 SWE-bench Verified issues and public per-instance outcomes for Claude 4.5 Haiku, Sonnet, and Opus under the same mini-SWE-agent v2.0.0 harness and reasoning setting. Repository families were held apart: 291 issues across six repositories trained the cards, 97 issues across two repositories selected the operating point, and 112 issues across four different repositories formed the test split. The classifier saw only the repository name and the first 2,000 characters of the issue, matching its runtime context limit

Rule-conditioned calibration improved held-out Brier score from 0.3047 to 0.1974, log loss from 0.9078 to 0.5829, and expected calibration error from 0.2362 to 0.0184. At the quality-first `0.95` objective weight, the validation-selected route matched the original cards at 80.36% test solve rate, while mean recorded cost changed from 0.7839 to 0.8263. A lower-cost point on the learned curve reached 77.68% solve rate at 0.6134 cost

The repository holdout exposed that model-wide calibration alone removed too much task discrimination. Attaching a smoothed outcome probability to the matched capability rule raised the learned `0.9`-weight route from 67.86% to 70.54% solve rate and from 0.7107 to 0.7204 utility. It still did not beat the original cards on test routing utility, so this result supports the calibration and evaluation machinery rather than a coding preset quality claim

### tau2-bench pipeline run

The pipeline was exercised against public tau2-bench trajectories for `claude-sonnet-4-5` and `claude-opus-4-5`, with four recorded end-to-end attempts aggregated per task. Entire domains were held apart: 50 airline tasks trained the artifact, 113 retail tasks selected the operating point, and 110 telecom tasks were evaluated once. A local `mlx-community/Qwen3-4B-Instruct-2507-4bit` model produced the capability forecasts

At a `0.7` quality weight, the validation-selected configuration reached 85.00% test solve rate at 0.4678 mean recorded cost. The original cards reached 92.27% at 0.7057 cost. This is 33.7% lower cost with a 7.27-point solve-rate loss, and improves the configured normalized utility from 0.6650 to 0.8950. The learned test curve also contains a 91.36% solve-rate point at 0.6743 cost, 4.4% below the original cost with a 0.91-point solve-rate loss

Calibration generalized across the held-out domain. Brier score improved from 0.2207 to 0.0687, log loss from 0.7657 to 0.4326, and expected calibration error from 0.2984 to 0.1627

This run validates artifact training, domain-disjoint evaluation, and the quality-cost tradeoff. It does not show a strict raw solve-rate improvement over the original cards, and it is not a direct comparison with a published Switchyard result. Larger cross-benchmark training data and another untouched test family are required before treating these cards as a general preset
