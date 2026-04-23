# Adaptive Router (v0)

A request-type-aware routing strategy. For each incoming request, classify the
prompt into one of seven `RequestType` buckets (code generation, writing,
analytical reasoning, …), then Thompson-sample a Beta(α, β) bandit posterior
per `(request_type, model)` cell to pick the best model. Quality estimates are
combined with a normalized cost score via a weighted linear sum.

A post-call hook reads the response and runs lightweight regex + tool-call
detectors (see `signals.py`) to award per-turn credit/blame to the model that
served the turn. Updates are batched in-memory and flushed to Postgres every
~10s by a background task in `proxy_server.py`.

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
```

Callers may pass header `x-litellm-min-quality-tier: 3` (or metadata key
`min_quality_tier: 3`) to force selection from tier-3-or-higher models only.

## Behavior summary

- **Cold start.** Each `(request_type, model)` cell starts with a
  Beta prior whose mean = `BASE_TIER_WEIGHT[tier] (+ STRENGTH_BONUS if declared)`
  and total mass = `COLD_START_MASS` (10). About ten real observations move it
  meaningfully.
- **Per-request decision.** Sample once per eligible model, score with
  `quality_weight·sample + cost_weight·normalized_cost`, pick the argmax.
  Routing is stateless per-turn — no sticky lookup. Each call resamples.
- **Owner-cache attribution.** Post-call, the conversation's first picked
  model claims an "owner slot" for `OWNER_CACHE_TTL_SECONDS` (24h). Later
  turns of the same conversation only fire bandit/state updates if the
  same model handled them — mismatches are dropped (no attribution) and
  counted in `skipped_updates_total`. Conversation identity is the
  client-supplied `litellm_session_id` if present, otherwise a sha256 over
  caller identity (api key hash, team, user, end-user) + the first message.
- **Per-turn updates.** `satisfaction → +α`. `misalignment, stagnation,
  disengagement, failure → +β` (each). `loop → +0.5β`. `exhaustion → 0`
  (uptime, not quality). Skipped if conversation has fewer than
  `SIGNAL_GATE_MIN_MESSAGES` messages.
- **Persistence.** Bandit cells: aggregated deltas, eventually consistent.
  Session rows: last-write-wins snapshots.

## Known v0 limitations

- **Latency is not in the score.** Quality + cost only. A pathologically slow
  model can still be picked.
- **Hard sample cap at 200.** Once `α + β > 200`, deltas are silently dropped.
  No rescaling — drift is a v1 concern.
- **24h owner-cache TTL.** No explicit eviction below TTL. The in-memory map
  can grow if traffic patterns produce many one-shot sessions.
- **Owner-recovery skew.** If model A "owns" a conversation but is then
  dethroned in the bandit, later turns served by model B are dropped — so
  bandit updates for that conversation flatline until A's TTL expires.
  Tracked via `skipped_updates_total`.
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
