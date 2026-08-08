# Phase 1: Efficiency signal in the adaptive-router bandit

## What this is, in one sentence

Give the bandit that already runs by default (`complexity_router` with `adaptive: true`) a second
posterior — keyed on latency/throughput, not just the behavioral-signal quality proxy it has
today — so a fast-but-wrong-tempo model and a slow-but-adequate model are distinguishable at
selection time. This is the one piece of the EURo package that doesn't route through an LLM judge
and doesn't touch classification, so it's safe to ship without opening either of those questions.

## Why now — the exact gap, cited

Every fact below was re-verified in this worktree before writing this plan, not carried over from
memory.

1. **The bandit today has no efficiency term at all.** `BanditCell` is `(alpha, beta)` — a single
   Beta posterior — and it only ever moves from behavioral signals detected in `signals.py`
   (misalignment / stagnation / disengagement / satisfaction / failure / loop / exhaustion). None
   of those signals look at latency or tokens/sec.
   [`adaptive_router/bandit.py:27-33`](litellm/router_strategy/adaptive_router/bandit.py:27),
   [`adaptive_router/adaptive_router.py:489-501`](litellm/router_strategy/adaptive_router/adaptive_router.py:489)

2. **The data is already flowing into the hook and being thrown away.** The post-call hook
   receives `kwargs, response_obj, start_time, end_time` — latency is `end_time - start_time`,
   completion tokens are on `response_obj.usage.completion_tokens` — and today it extracts none of
   that; it only reads `kwargs["messages"]` and `response_obj` for text.
   [`adaptive_router/hooks.py:199-251`](litellm/router_strategy/adaptive_router/hooks.py:199)
   Compare to `lowest_latency.py`, which already does exactly this extraction for a different
   strategy — `_usage.completion_tokens`, `response_ms = end_time - start_time`
   ([`lowest_latency.py:83-95`](litellm/router_strategy/lowest_latency.py:83)) — so this is not new
   plumbing, it's reusing a pattern that ships today, in a different file.

3. **This is why static routing beats the bandit on anything speed-sensitive.** Selection scores
   `quality_weight * thompson_sample(quality) + cost_weight * normalized_cost(...)`
   ([`adaptive_router.py:97-109`](litellm/router_strategy/adaptive_router/adaptive_router.py:97)).
   `cost` here is the *declared $/1k-token price* in config, a static number — it cannot reflect
   that a model degrades in practice (rate-limited, overloaded, verbose). Two models with identical
   quality posteriors and identical prices are indistinguishable today even if one is consistently
   3x slower under load. Static per-tier pinning can't do better here either, but at least an
   operator can eyeball latency dashboards and repin manually. The bandit currently can't correct
   for this on its own, which undercuts its own pitch (adapt automatically).

4. **This is the one piece of the reference package that's classifier-agnostic and judge-free**
   (established in the prior discussion this turn, and re-confirmed above): EURo's composite
   formula is pure arithmetic on `latency_seconds` and `completion_tokens`, no model call. It
   doesn't require `classifier_type: "llm"`, doesn't touch `keyword_tier_rules`, doesn't need a
   judge. Every other candidate improvement (hierarchical tier cells, judge-scored quality) either
   needs a design review or an LLM in the loop; this one needs neither, which is why it's Phase 1.

## What "done" looks like

- A `(request_type, model)` cell tracks a second Beta posterior, `alpha_eff`/`beta_eff`, fed by
  `reward = gamma * exp(-latency / target_latency) + (1 - gamma) * min(tokens/latency/throughput_max, 1)`,
  ported from the reference package's `efficiency_router.py` reward formula, clamped to `[0, 1]`.
- Selection blends three terms instead of two:
  `score = quality_weight * TS(quality) + efficiency_weight * TS(efficiency) + cost_weight * normalized_cost(...)`.
- The new weight defaults to something that **cannot silently change existing routing behavior**
  for operators who don't opt in (see Decision 1 below) — this is a hard constraint on the design,
  not a nice-to-have.
- Persisted the same way the quality cell already is: Prisma columns + `AdaptiveRouterStateRepository`,
  batched through the same `AdaptiveRouterUpdateQueue`, loaded on the same `load_state_from_db` path.
- `get_state_snapshot()` reports the efficiency mean alongside `quality_mean` — no new endpoint.

## Decisions that need to be made explicit before writing code

These are genuine judgment calls, not implementation detail. I'm stating my recommendation and the
reasoning for each; flag which ones you want a different default in.

**Decision 1 — does this change behavior for existing `adaptive: true` deployments by default?**
`AdaptiveRouterWeights` currently validates `quality + cost == 1.0`
([`types/router.py:894-900`](litellm/types/router.py:894)) — a *closed* two-term simplex. Adding a
third term means either (a) extending the validator to `quality + cost + efficiency == 1.0` with
`efficiency` defaulting to `0.0`, so nothing changes unless an operator explicitly reallocates
weight into it, or (b) defaulting it to something nonzero (e.g. `0.15`, taken proportionally from
`cost`) so the feature does something out of the box.
**Recommendation: (a), default `0.0`.** An operator who already tuned `quality`/`cost` for their
fleet did so under a model where the score has two terms; silently inserting a third and
renormalizing changes their routing without their input. Opt-in first, propose a nonzero default
once we have even one deployment's before/after data. This mirrors how `complexity_router`
shipped `session_affinity` off by default for an analogous reason
([`complexity_router/config.py:443-451`](litellm/router_strategy/complexity_router/config.py:443)).

**Decision 2 — where does `target_latency` come from?**
EURo's reference calibrates it once at startup by timing the smallest model
(`test_custom_router_euro_v1.py: calibrate_target_tpt`). We don't have a "smallest model" concept —
`adaptive_router` treats its `available_models` as peers. Two options: (i) a static config default
(e.g. `1.0`s, same as EURo's) that operators override per deployment, or (ii) auto-calibrate from
the first N observed latencies per model group, adaptively.
**Recommendation: (i) for Phase 1.** Auto-calibration is real complexity (needs a warm-up window,
needs to handle cold restarts, needs to not be gamed by the first slow request) and is explicitly
scoped as Phase 2 in the broader roadmap already discussed. Shipping a static, override-able
default now and layering calibration on top later is strictly additive — it won't require
reworking Phase 1's schema.

**Decision 3 — reward on failure.**
EURo's convention: reward = 0 on failure/timeout. `record_turn` already receives
`turn.response_status` and threads it through `failure` detection for the quality signal
([`hooks.py:204`](litellm/router_strategy/adaptive_router/hooks.py:204),
[`signals.py`](litellm/router_strategy/adaptive_router/signals.py)). **Recommendation:** reuse
that same `response_status` to zero the efficiency reward on non-2xx, rather than inventing a
second failure-detection path. One source of truth for "did this call fail."

**Decision 4 — migration shape.**
Extend `LiteLLM_AdaptiveRouterState` in place (`alpha_eff FLOAT @default(1.0)`,
`beta_eff FLOAT @default(1.0)`) versus a new sibling table. **Recommendation:** extend in place.
The row is already keyed `(router_name, request_type, model_name)` — exactly the cell efficiency
needs — and a sibling table would require a second `find_many` + a join to reconstitute one
logical cell on every `load_state_from_db` call for no isolation benefit (nothing reads efficiency
independently of quality).

## Scope boundary — explicitly NOT part of Phase 1

- No change to `classifier_type` default or to any classification code path.
- No hierarchical `(tier, request_type, model)` cell — that's the separate, harder change
  (needs a shrinkage design) already flagged as Phase 3 in the prior discussion.
- No judge / LLM-scored quality anywhere. Confirmed in the prior turn: the judge is what drives
  EURo's *accuracy* claim on deceptively-simple prompts; efficiency reward alone cannot detect a
  wrong-but-fast answer. Phase 1 does not claim to fix that class of misrouting — it only fixes
  "the bandit can't tell fast-adequate from slow-adequate."
- No auto-calibration of `target_latency` (Decision 2 above) — static default only.
- No change to `complexity_router`'s non-adaptive (`random.choice`) path — this only touches the
  bandit cells that `adaptive: true` deployments already read from.

## Implementation checklist

1. `litellm/types/router.py` — extend `AdaptiveRouterWeights` with `efficiency: float = 0.0` and
   widen the sum-to-one validator to three terms.
2. `litellm/router_strategy/adaptive_router/config.py` — port constants: `DEFAULT_EFFICIENCY_WEIGHT`,
   `GAMMA`, `DEFAULT_TARGET_LATENCY`, `THROUGHPUT_MAX` (mirror EURo's `efficiency_router.py`
   defaults: `gamma=0.5`, `target_latency=1.0`, `throughput_max=100.0`), all marked `UNVALIDATED`
   per the file's existing convention.
3. `litellm/router_strategy/adaptive_router/bandit.py` — add `alpha_eff`/`beta_eff` fields to
   `BanditCell` (default `1.0`/`1.0`, uninformative prior, matching the existing quality-cell
   convention of `Beta(1,1)` before `initial_cell()` biases it); add `thompson_sample_efficiency`
   (or parameterize the existing `thompson_sample`); extend `pick_best`/`score` to blend the third
   term.
4. New `litellm/router_strategy/adaptive_router/efficiency.py` — the reward function, ported and
   attributed: `composite_efficiency_reward(latency_seconds, completion_tokens, gamma,
   target_latency, throughput_max) -> float`, plus a docstring crediting the source formula the
   same way other ported logic in this codebase cites its origin.
5. `litellm/router_strategy/adaptive_router/hooks.py` — in `_record`, extract `end_time - start_time`
   and `response_obj.usage.completion_tokens` (mirroring `lowest_latency.py:83-95`'s
   `timedelta`-normalization guard), zero the reward on `response_status != 2xx` (Decision 3), call
   a new `AdaptiveRouter.record_efficiency(...)` alongside the existing `record_turn(...)`.
6. `litellm/router_strategy/adaptive_router/adaptive_router.py` — `record_efficiency()` method,
   queues an `alpha_eff`/`beta_eff` delta through the existing `AdaptiveRouterUpdateQueue` (extend
   its aggregation payload, don't add a second queue).
7. `litellm/router_strategy/adaptive_router/update_queue.py` — extend the state-delta payload shape
   to carry `delta_alpha_eff`/`delta_beta_eff` alongside the existing `delta_alpha`/`delta_beta`.
8. `schema.prisma` — add `alpha_eff`/`beta_eff` columns to `LiteLLM_AdaptiveRouterState` (Decision 4);
   generate the migration the way this repo already does for Prisma changes.
9. `adaptive_router.py: load_state_from_db` — read the two new columns into `BanditCell`.
10. `adaptive_router.py: get_state_snapshot` — report `efficiency_mean` per cell.
11. Tests — extend, don't replace, the existing suite:
    `test_bandit.py` (new cell fields + blended scoring), `test_hooks.py` (latency/token extraction,
    failure zeroing), `test_update_queue.py` (extended payload aggregates correctly),
    `test_adaptive_router.py` (record_efficiency wiring), `test_state_endpoint.py` (snapshot
    includes efficiency_mean). No new test files needed — this is additive to an existing,
    well-organized suite.
12. Config docs — extend `adaptive_router/README.md`'s config example with the new weight and
    reward constants, same section structure it already uses.

## How I'll know Phase 1 actually helped (not just shipped)

Per Risk 6 from the earlier analysis, neither existing eval harness is trustworthy for this:
`eval_complexity_router.py` measures classification accuracy against fuzzy bands, not routing
outcomes, and EURo's own prompt set is constructed to make static routing lose. Before calling
Phase 1 validated:

- Build a strict-scoring replay harness over real (redacted) production routing-decision logs —
  `StandardLoggingRoutingDecision` already records `tier`, `request_type`, `routed_model` per
  request ([`types/utils.py:2785-2804`](litellm/types/utils.py:2785)), so latency/tokens can be
  joined from the same spend-log row without new instrumentation.
- Compare `efficiency_weight=0` (today's behavior) against a nonzero value on replayed traffic,
  holding quality/cost weights fixed, and look at p50/p95 latency shift with quality held constant
  (no regression in behavioral-signal-derived quality mean).
- Only then propose a nonzero default (revisit Decision 1).
