# Adaptive Router (v0.1)

A request-type-aware routing strategy. For each incoming request, classify the
prompt into one of seven `RequestType` buckets (code generation, writing,
analytical reasoning, …), then Thompson-sample a Beta(α, β) bandit posterior
per `(request_type, model)` cell to pick the best model. Quality estimates,
efficiency estimates (latency + throughput), and normalized cost are combined
via a weighted linear sum.

A post-call hook reads the response and runs lightweight regex + tool-call
detectors (see `signals.py`) to award per-turn credit/blame to the model that
served the turn. Independently, it extracts latency and token count to update
an efficiency posterior (see `efficiency.py`). Updates are batched in-memory
and flushed to Postgres every ~10s by a background task in `proxy_server.py`.

## Config example

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
    model_info:
      input_cost_per_token: 0.0000025
      adaptive_router_preferences:
        quality_tier: 3
        strengths: ["code_generation", "analytical_reasoning"]

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
    model_info:
      input_cost_per_token: 0.00000015
      adaptive_router_preferences:
        quality_tier: 2
        strengths: ["general", "factual_lookup"]

  - model_name: smart-router
    litellm_params:
      model: auto_router/adaptive_router
      adaptive_router_default_model: gpt-4o-mini
      adaptive_router_config:
        available_models: ["gpt-4o", "gpt-4o-mini"]
        weights:
          quality: 0.7
          cost: 0.3
          efficiency: 0.0  # Optional: set to >0 to factor latency/throughput into selection
```

Callers may pass header `x-litellm-min-quality-tier: 3` (or metadata key
`min_quality_tier: 3`) to force selection from tier-3-or-higher models only.

## Behavior summary

- **Cold start.** Each `(request_type, model)` cell starts with a
  Beta prior whose mean = `BASE_TIER_WEIGHT[tier] (+ STRENGTH_BONUS if declared)`
  and total mass = `COLD_START_MASS` (10). About ten real observations move it
  meaningfully.
- **Per-request decision.** Sample once per eligible model, score with
  `quality_weight·TS(quality) + efficiency_weight·TS(efficiency) + cost_weight·normalized_cost`,
  pick the argmax. Efficiency is `gamma·exp(-latency/target) + (1-gamma)·min(tokens/sec/max, 1)`;
  see `efficiency.py`. Routing is stateless per-turn — no sticky lookup. Each call resamples.
- **Previous-response attribution.** Post-call, feedback from the current user
  message is attributed to the model that produced the previous response, while
  response signals are attributed to the current model. Contexts expire after
  24 hours and the in-memory cache is capped at 1,024 sessions. Conversation identity is the
  client-supplied `litellm_session_id` if present, otherwise a sha256 over
  caller identity (api key hash, team, user, end-user) + the first message.
- **Per-turn quality updates.** `satisfaction → +α`. `misalignment, stagnation,
  disengagement, failure → +β` (each). `loop → +0.5β`. `exhaustion → 0`
  (uptime, not quality). Skipped if conversation has fewer than
  `SIGNAL_GATE_MIN_MESSAGES` messages.
- **Per-response efficiency updates.** Every completion updates the efficiency
  posterior independently: `reward = gamma·exp(-latency/target_latency) + (1-gamma)·throughput`,
  then `+α_eff = reward`, `+β_eff = 1 - reward`. No cold-start prior; starts at Beta(1, 1).
  On failure (non-2xx), `reward = 0` (same failure detection as quality signals).
- **Persistence.** Bandit cells: aggregated deltas, eventually consistent.
  Both quality and efficiency posteriors live in the same row in `LiteLLM_AdaptiveRouterState`.
  Session rows: last-write-wins snapshots.

## Efficiency weighting (v0.1+)

The `efficiency` weight is **optional and defaults to 0.0** (off by default).
When set to a value > 0, it reallocates from the `quality` + `cost` budget:
the three weights must still sum to 1.0. Operators should start with a small
value (e.g., `efficiency: 0.1`) and tune based on their workload's latency
sensitivity. No auto-calibration yet; `target_latency` is a static 1.0s by default
and can be overridden in config.

## Known v0.1 limitations

- **Hard sample cap at 200.** Once `α + β > 200` (quality or efficiency), deltas
  are silently dropped. No rescaling — drift is a v1 concern. This cap is shared
  between quality and efficiency on the same cell.
- **No auto-calibration.** `target_latency` is static (1.0s default). Phase 2
  will add a warm-up window to auto-detect per model group.
- **Signals are regex + tool-call only.** No LLM-judge, no embedding similarity,
  no exemplar storage. Signals are best-effort and biased toward English.
- **One AdaptiveRouter per `Router`.** Multiple `adaptive_router/*` deployments
  on the same `litellm.Router` raise at init.
- **Bandit-delta mapping is unvalidated.** `_compute_bandit_delta` is a v0
  guess; expect to retune after the first ~1000 sessions of real traffic.
- **`request_type` is classified per turn from the latest user message.** For
  non-GENERAL turns, the current-turn type is used for bandit attribution (so
  genuine mid-session topic shifts update the correct cell). For GENERAL turns
  ("thanks!", "ok", "sounds good"), attribution falls back to the session's
  original type to avoid misattributing closing pleasantries.
