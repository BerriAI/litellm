# Capability router

The capability router asks an LLM to forecast each configured candidate's probability of completing the current task, then selects the cheapest candidate whose forecast clears the configured reliability threshold. This is separate from the complexity router and its local or trained heuristic classifiers

## Train capability cards

Capability-card training uses outcomes from the real task harness. It does not replace the LLM classifier. The classifier still reads the task and cards at runtime; the trained artifact improves the evidence and policy around its forecast by:

- assigning each card rule a boundary from observed end-to-end outcomes and a 95% Wilson interval;
- fitting monotonic probability calibration for each candidate;
- tuning the global probability threshold and boundary step on validation tasks; and
- reporting raw and calibrated Brier score, log loss, calibration error, solve rate, cost, and quality-cost utility on untouched test tasks

Create one JSONL row for every `(task, candidate, run)`:

```json
{"benchmark":"terminal-bench-2.1","task_id":"build-linux-kernel-qemu","split":"train","model":"efficient","primary_rule":"R3","raw_p_solve":0.72,"success":1.0,"estimated_cost":0.18}
```

`raw_p_solve` and `primary_rule` come from the capability classifier's routing decision. `success` must come from the benchmark's end-to-end verifier, not an LLM estimate. Repeated runs may use the same task and model; the trainer averages them when measuring routing quality

Every benchmark task must have an explicit `train`, `validation`, or `test` split and outcomes for every configured candidate. A task cannot cross splits. For a general preset, assign entire benchmark families to a split and deduplicate related tasks before training. A random row split over near-duplicate tasks overstates generalization

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
