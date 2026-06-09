# Governor: budget, rate-limit, and quota enforcement (v2)

Governor is a parallel implementation of LiteLLM's per-request governance: spend tracking, budgets, RPM/TPM and concurrent-request limits, cost-based throttling, and an audit log of every admit/reject decision. It is inert until `LITELLM_GOVERNOR_V2` is truthy; when off, the legacy hooks under `litellm/proxy/spend_tracking/` and `litellm/proxy/hooks/` are the active path and this package does nothing. Governor does NOT replace authentication, does NOT compute model pricing (it reads costs the rest of the stack produces), and does NOT own the SpendLog/LedgerEvent durable store; it borrows the existing Prisma models and the existing `pod_lock_manager` for background-job leadership.

## Top-level structure

Mirrors `litellm/integrations/otel/`. Closed-set layering: `model/` imports nothing outside itself plus `litellm._logging`; `plumbing/` imports `model/`; `policies/` imports `model/` and `plumbing/`; `engine/` imports all three; `adapter/`, `audit/`, and `presets/` import the engine.

```
litellm/integrations/governor/
  __init__.py
  README.md                       (written once code lands; this DESIGN.md is the planning artifact)
  runtime.py                      SDK-free entrypoints proxy core may call before settings load
  model/
    __init__.py
    config.py                     GovernorV2Config (pydantic-settings BaseSettings); is_governor_v2_enabled()
    subjects.py                   Subject, SubjectKind, SubjectRef
    decisions.py                  Decision, Verdict, RejectReason, FailMode, Status enums
    policies.py                   PolicyConfig dataclasses (BudgetPolicyConfig, RpmPolicyConfig, ...)
    counters.py                   Counter, BudgetWindow, RateLimitWindow, GcraState, Outcome
    audit.py                      AuditEvent dataclass + AuditSeverity enum
    keys.py                       Pure functions: counter_key(subject, policy_id, window_bucket) -> str
  plumbing/
    __init__.py
    cache.py                      ThreeTierCache facade. Composes L1 + L2 + L3.
    inmemory.py                   BoundedLRUCounterCache (asyncio.Lock-guarded LRU)
    redis.py                      RedisCounterStore + registered Lua scripts
    postgres.py                   PostgresCounterStore (Prisma-backed). Idempotent batched flush.
    locks.py                      Re-export of pod_lock_manager helpers; no new lock impl
    lua/                          *.lua files loaded at module import (one script per file)
      check_and_increment.lua     The single atomic primitive (see section 5)
      gcra_admit.lua              GCRA token-bucket admission (vendored from redis-cell, NOT a runtime dep)
      reconcile_actual.lua        Post-call delta application
    clock.py                      Injectable time source. Lua scripts use redis TIME.
  policies/
    __init__.py
    base.py                       Policy Protocol + AdmitContext, ReconcileContext
    budget.py                     BudgetPolicy (window-bucketed cost counter; fail-closed default)
    rpm.py                        RpmPolicy (GCRA on request count; fail-open default)
    tpm.py                        TpmPolicy (sliding-window-counter on token count; fail-open default)
    concurrent.py                 ConcurrentRequestPolicy (sorted-set inflight, fail-closed default)
    cost_throttle.py              CostThrottlePolicy (delay/reject as fraction-of-budget rises)
    registry.py                   Built-in policy registry (str -> class)
  engine/
    __init__.py
    governor.py                   Engine: load_policies(), admit(ctx) -> Decision, reconcile(ctx)
    admission.py                  Pure function: combine per-policy verdicts (worst-wins) -> Decision
    degradation.py                FailMode dispatch (engine-owned, not policy-owned)
    workers.py                    Background tasks: window-reset scheduler, L2->L3 flusher
  adapter/
    __init__.py
    governor_v2.py                GovernorV2(CustomLogger): wires admit/reconcile to proxy hooks
    response.py                   Decision -> 429 / 403 response + headers
  audit/
    __init__.py
    emitter.py                    AuditEmitter Protocol + dispatch
    sinks/
      prisma.py                   Append to LiteLLM_GovernanceAuditLog table (new)
      verbose_logger.py           Structured emit via litellm._logging
      otel.py                     Optional: forward to otelv2 as span events / OTel Logs API
  presets/
    __init__.py                   PRESET_BY_NAME registry
    typical_key_budget.py
    typical_team_rpm.py
    enterprise_full.py
```

The split rationale is the same as otelv2's: pure data types stay portable and import-cheap; engine orchestration is one file you can read end-to-end; adapter and presets are the only spots that know LiteLLM exists.

## Dependencies

Governor adds no required runtime dependencies. It uses what LiteLLM already imports: `redis.asyncio`, prisma, `pydantic-settings`, the stdlib.

Optional opt-ins, picked up only when present:

- `aiolimiter` for an in-process per-pod token bucket on the L1 shaper. Useful for "soft" RPM limits where Redis round-trips per admission are too expensive (e.g. ultra-high-RPS keys). Adopted as Tier-1 hot path only; never as cross-pod consensus.
- `limits` for the prototype phase only; the moving/sliding-counter strategies are referenced when designing the TPM policy. Not a runtime dependency.

Vendored, not depended on:

- The GCRA algorithm pattern (originally from `redis-cell`). The `redis-cell` Rust module is impractical on managed Redis (ElastiCache, MemoryDB) since it requires loading a custom `.so` into the server. The algorithm is small and well-documented; vendoring keeps governor portable.

## Data model

Pure types live in `model/`. Every type is `@dataclass(frozen=True)` unless it carries mutable state (`Counter.value` reads from a tier; the tier owns mutation). Enums are `Literal[...]` aliases when they are pure tags, `enum.Enum` when they need methods.

```python
SubjectKind = Literal["key", "team", "user", "end_user", "tag", "org", "model"]
FailMode    = Literal["closed", "open"]
Status      = Literal["admitted", "admitted_degraded", "rejected"]
WindowKind  = Literal["daily", "weekly", "monthly", "rolling"]

@dataclass(frozen=True)
class Subject:
    kind: SubjectKind
    id: str                       # opaque; for "tag" this is the tag value, for "key" the key hash
    display: str | None = None    # for audit; not used in counter keys

@dataclass(frozen=True)
class SubjectRef:
    subjects: tuple[Subject, ...] # the hierarchy in priority order: end_user, key, team, user, org

class AdmitContext(TypedDict):
    request_id: str
    subjects: SubjectRef
    model: str
    estimated_cost: float | None  # None when cost is unknown (e.g. image route)
    estimated_tokens: int | None
    metadata: Mapping[str, str]

@dataclass(frozen=True)
class Verdict:
    policy_id: str
    status: Status
    reason: str | None
    limit: float | None
    observed: float | None
    fail_mode: FailMode
    degraded: bool                # True when the tier outage forced fail-open

@dataclass(frozen=True)
class Decision:
    status: Status
    verdicts: tuple[Verdict, ...]
    request_id: str
    latency_ms: float
    @property
    def rejected_by(self) -> tuple[Verdict, ...]: ...
    @property
    def headers(self) -> Mapping[str, str]: ...  # X-RateLimit-* and X-Budget-* hints

@dataclass(frozen=True)
class BudgetWindow:
    kind: WindowKind
    period: timedelta
    bucket_key: str               # e.g. "2026-06-09" for daily; deterministic from kind + now
    @classmethod
    def current(cls, kind: WindowKind, now: datetime) -> "BudgetWindow": ...

@dataclass(frozen=True)
class RateLimitWindow:
    period_seconds: int
    capacity: int                 # max events per window
    algorithm: Literal["gcra", "sliding_window"]

@dataclass(frozen=True)
class Counter:
    kind: Literal["spend", "rpm", "tpm", "inflight"]
    subject: Subject
    policy_id: str
    bucket_key: str               # window bucket for budgets; rolling-window tag for rates
    value: float
    ttl_seconds: int

@dataclass(frozen=True)
class Outcome:
    success: bool
    actual_cost: float
    actual_input_tokens: int
    actual_output_tokens: int
    error_class: str | None
    upstream_status: int | None

@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    decision: Decision
    subjects: SubjectRef
    occurred_at: datetime
    latency_ms: float
    outcome: Outcome | None       # populated on reconcile; None on bare admit
    correlation: Mapping[str, str]  # trace_id, span_id (when otelv2 is on), call_id
```

`Policy` is a Protocol, not an ABC; that keeps the policy types pluggable without forcing inheritance and reads cleanly under `mypy --strict`:

```python
class Policy(Protocol):
    policy_id: str
    fail_mode: FailMode
    enabled: bool
    async def admit(self, ctx: AdmitContext, cache: ThreeTierCache) -> Verdict: ...
    async def reconcile(self, ctx: AdmitContext, outcome: Outcome, cache: ThreeTierCache) -> None: ...
```

Policies do NOT decide what happens on tier outage; they raise typed `TierDegraded` and `engine/degradation.py` converts that into either a `Reject` or an `Admit-Degraded` verdict per the policy's declared `fail_mode`. This is the single most important architectural rule and is enforced by Verdict construction (only the degradation module can set `degraded=True`).

## Hierarchical evaluation: Envoy descriptor model

Governance follows Envoy's RLS descriptor pattern. A request carries a descriptor tuple ordered by specificity (`end_user`, `key`, `team`, `user`, `tag*`, `org`); each policy declares which subject it governs; the engine evaluates every applicable policy and rejects if ANY verdict is over-limit. This is straight from Envoy's data model and matches Stripe's hierarchical rate-limit pattern; we do not depend on Envoy's Go ratelimit service. The descriptor tuple is stable and ordered so a request from the same end_user under the same key always produces the same evaluation order, which is necessary for the audit trail to be useful for debugging "why was this admitted/rejected?".

## Cache tiering contract

Three tiers; only the cache facade speaks to them. Policies read and write `Counter` values through `ThreeTierCache`; they never reach a Redis client.

### L1 in-memory (per pod, bounded LRU)

Hot read path. An asyncio-locked `OrderedDict` capped at a configured size (default 16k entries). The L1 entry holds the last-known counter value plus the timestamp it was read from L2; reads under `staleness_ms` return L1 immediately, older reads probe L2. L1 is write-through-on-read: nothing outside the cache facade ever calls `l1.set`; the facade populates L1 after every successful L2 read or increment. L1 is wiped on process restart, by design.

L1 is bypassed for high-cardinality subjects (`end_user`, `tag`) to avoid thrashing. A team with 100k end-users would otherwise churn the LRU on every admission; for those subjects the cache goes straight to L2 every time.

Consistency vs L2: eventual, bounded by `staleness_ms` (default 250ms for rates, 1s for budgets). A pod that misses an increment that happened on another pod sees the new value within `staleness_ms` of the next admit on that subject.

Failure mode: L1 can never fail in a way that matters (the LRU is in-memory). A capacity miss falls through to L2 cleanly.

### L2 Redis (cross-pod consensus authority)

The current-window counters live here. Atomic increments via Lua (see section 5). Per-policy TTL: budget counters get `window_period + reset_grace`, rate counters get a small fixed TTL so abandoned buckets evict.

Consistency vs L3: eventual; L3 is updated by a background flusher (section 7). L2 is the source of truth for *the current window only*; once a window closes, its counter ages out and L3 holds the historical sum.

Failure mode: every operation is wrapped in `RedisCounterStore.with_retry`, which retries once with 50ms jitter on `RedisError` or `TimeoutError` and otherwise raises `TierDegraded(reason="redis_unavailable", tier="L2")`. The wrapper NEVER catches and returns 0, NEVER deletes a key on error, and NEVER swallows an exception silently.

### L3 Postgres (durable truth)

The Prisma database. Holds the canonical historical spend per subject and the configuration tables (budgets, policy bindings). Reads happen on warm start; writes happen via the flusher, never on the hot path. Reservation-style flushes are idempotent: each batch carries a deduplication token (request_id + policy_id) recorded in `LiteLLM_GovernanceLedger`, and the flusher applies only entries not yet present. This is the regression-test surface for the spend-counter bug class: the flusher cannot widen a single Redis blip into permanent under-counting.

Consistency vs L2: eventual; the flusher reconciles every `flush_interval_seconds` (default 5s) and at window boundaries. If a flush fails, the batch sits in an in-memory append-only `PendingFlushQueue` (bounded) and retries on the next tick; nothing in L2 is touched. The pod_lock_manager ensures only one pod runs the flusher at a time.

Failure mode: if Postgres is down, the flusher backs off and queues. Admission continues from L2 unaffected. If the queue overflows (Postgres outage longer than ~10 minutes at default sizing), the oldest entries are dropped to an on-disk WAL file (`./governor_pending_flush.log`) and the eviction is logged at WARNING.

### Failure flow under each policy fail-mode

| Tier outage | Fail-closed policy (e.g. BudgetPolicy) | Fail-open policy (e.g. RpmPolicy) |
| --- | --- | --- |
| L1 LRU full | transparent; falls through to L2 | transparent; falls through to L2 |
| L2 Redis timeout | Verdict(status=rejected, reason="degraded_fail_closed", degraded=True) | Verdict(status=admitted_degraded, reason="degraded_fail_open", degraded=True) |
| L3 Postgres down | admission unaffected; flusher queues. If queue overflows AND fail-closed budgets exist, engine emits CRITICAL audit; admission continues from L2 | admission unaffected |

The engine, not the policy, picks the row. A policy author writes only the happy path; degradation is mechanical.

## The atomic counter primitive

One Lua script, `check_and_increment.lua`, loaded once at startup via `EVALSHA`. Its properties are non-negotiable:

```lua
-- KEYS[1] = counter_key
-- KEYS[2] = window_key (optional; absent for rolling-only counters)
-- ARGV[1] = limit                  (float; -1 disables cap)
-- ARGV[2] = increment              (float)
-- ARGV[3] = window_period_seconds  (int; 0 = no window reset semantics)
-- ARGV[4] = counter_ttl_seconds    (int; 0 = no TTL change)
-- Returns:
--   { 0, new_value, limit, ttl_remaining }   on success
--   { 1, current_value, limit, ttl_remaining } on over-limit (NO mutation)
--   { 2, "<lua error tag>" }                  on internal Lua error (NO mutation)

local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])

local limit         = tonumber(ARGV[1])
local increment     = tonumber(ARGV[2])
local window_period = tonumber(ARGV[3])
local ttl           = tonumber(ARGV[4])
local counter_key   = KEYS[1]

local current = tonumber(redis.call('GET', counter_key) or '0')

-- Window-reset is consulted via window_key + window_period. A reset is
-- only triggered when the window has elapsed; on reset, the counter starts
-- from `increment`, not from 0. We NEVER call DEL on the counter key.
if window_period > 0 and #KEYS >= 2 then
    local window_key = KEYS[2]
    local window_start = redis.call('GET', window_key)
    if (not window_start) or (now - tonumber(window_start)) >= window_period then
        if limit >= 0 and increment > limit then
            return { 1, 0, limit, window_period }
        end
        redis.call('SET', counter_key, increment)
        redis.call('SET', window_key, tostring(now))
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
            redis.call('EXPIRE', window_key, ttl)
        end
        return { 0, increment, limit, ttl }
    end
end

if limit >= 0 and (current + increment) > limit then
    return { 1, current, limit, redis.call('TTL', counter_key) }
end

local new_value = redis.call('INCRBYFLOAT', counter_key, increment)
local current_ttl = redis.call('TTL', counter_key)
if ttl > 0 and current_ttl == -1 then
    redis.call('EXPIRE', counter_key, ttl)
end
return { 0, tonumber(new_value), limit, current_ttl }
```

Properties this enforces:

1. Read, check, increment are one EVAL.
2. Over-limit returns rejection without mutation.
3. `DEL counter_key` appears nowhere in any script.
4. Counter deletion is via TTL only, plus one explicit admin endpoint that always logs an `AuditEvent(severity=ADMIN_RESET)`.

The Python wrapper:

```python
class RedisCounterStore:
    async def check_and_increment(
        self, counter_key: str, window_key: str | None, *,
        limit: float, increment: float,
        window_period_s: int, ttl_s: int,
    ) -> CheckIncrementResult:
        try:
            result = await self._eval_with_retry(_CHECK_INCREMENT_SHA, ...)
        except (RedisError, TimeoutError) as e:
            raise TierDegraded(tier="L2", reason="redis_unavailable") from e
        return _parse_check_increment(result)
```

`_eval_with_retry` retries once on transient timeouts only (`asyncio.TimeoutError`, `ConnectionError`); on the second failure it raises `TierDegraded` and the engine takes over. The wrapper has no `except Exception:` that returns 0. There is no DEL path. There is no fallback that "reseeds from a snapshot column," which was the precise vector of the prior spend-counter bug.

## Rate-limit algorithm choice

- **RPM**: GCRA, vendored from the redis-cell algorithm. One TAT key per scope, atomic, burst-configurable, no background cleanup, smooths traffic. Never deletes a key on the decision path. GCRA is the simplest correct cross-pod RPM algorithm for this workload.
- **TPM**: sliding-window-counter. GCRA does not fit TPM because the increment (output tokens) is unknown at admission time; you'd be reserving against the upper bound on every request and refunding the delta at reconcile, which is more round-trips than just keeping a sliding window. Sliding-window-counter (two adjacent fixed buckets with linear interpolation) is well-known, cheap in Redis, and accurate enough for token rate limiting where the unit cost is fungible.
- **Concurrent requests**: not a rate problem. A concurrency gauge: INCR on admit / DECR on completion, plus a per-slot TTL safety net so crashed pods auto-release leaked slots. Reserve-and-commit/refund semantics, never DEL on error.

## Policy fail-mode plumbing

The engine sequences admission as:

```python
async def admit(self, ctx: AdmitContext) -> Decision:
    started = self._clock.now()
    verdicts: list[Verdict] = []
    for policy in self.policies_for(ctx.subjects):
        try:
            v = await policy.admit(ctx, self.cache)
        except TierDegraded as td:
            v = degradation.verdict_for_degradation(policy, td)
        verdicts.append(v)
    return admission.combine(verdicts, request_id=ctx.request_id,
                             latency_ms=(self._clock.now() - started).total_seconds() * 1000)
```

`degradation.verdict_for_degradation(policy, td)` returns:

- `Verdict(status="rejected", reason=f"degraded_fail_closed:{td.reason}", degraded=True)` when `policy.fail_mode == "closed"`;
- `Verdict(status="admitted_degraded", reason=f"degraded_fail_open:{td.reason}", degraded=True)` when `policy.fail_mode == "open"`.

`admission.combine` is worst-wins (any `rejected` -> overall reject, else any `admitted_degraded` -> overall admitted_degraded, else admitted). Reasons are concatenated; per-verdict `degraded` flags are preserved for the audit.

Concretely for the two canonical scenarios:

- Fail-closed `BudgetPolicy` with Redis timeout: policy raises `TierDegraded(L2)` -> `Verdict.rejected(degraded=True)` -> `Decision.rejected` -> adapter maps to HTTP 429 with `X-RateLimit-Reason: budget_unavailable_fail_closed`.
- Fail-open `RpmPolicy` with same: policy raises `TierDegraded(L2)` -> `Verdict.admitted_degraded(degraded=True)` -> `Decision.admitted_degraded` -> adapter forwards the request, sets `X-RateLimit-Degraded: true` on the response.

## Proxy integration points

### Pre-call admission

`GovernorV2(CustomLogger).async_pre_call_hook(user_api_key_dict, cache, data, call_type)` builds an `AdmitContext` from `user_api_key_dict`, the route, and an `estimated_cost` from the existing `estimate_request_max_cost` helper (the only piece of `litellm/proxy/spend_tracking/budget_reservation.py` we reuse). It calls `governor.admit(ctx)`. On `Decision.status == "rejected"`, the adapter raises `ProxyRateLimitError` (the existing exception class) with `code` derived from the worst verdict (`BUDGET_EXCEEDED`, `RPM_EXCEEDED`, `TPM_EXCEEDED`, `CONCURRENT_REQUESTS_EXCEEDED`, `BUDGET_UNAVAILABLE`). FastAPI exception handlers translate that to a 429 (or 403 for `BUDGET_EXCEEDED`, matching legacy behavior). Response headers stamped from `Decision.headers`.

### Post-call reconcile

`async_log_success_event` and `async_log_failure_event` build `Outcome` from the typed `standard_logging_object` (cost, prompt/completion tokens, error class) and call `governor.reconcile(ctx, outcome)`. Each policy applies its delta: `BudgetPolicy.reconcile` computes `actual_cost - estimated_cost` and applies via a second EVAL of the same `check_and_increment.lua` with `limit=-1` (no cap), which is idempotent against double-fire because both success and failure callbacks pass the same request_id and the wrapper records a small bloom over recent request ids per-policy to drop duplicates within a 60s window. (Cheaper than another round trip to Redis on every call.)

L3 flushes happen on a separate cadence (`workers.L3Flusher`), never inline with reconcile.

### Background workers

Three asyncio tasks owned by the engine, started once at proxy startup, all gated on `pod_lock_manager.acquire_lock(...)` so only one pod runs each:

1. `WindowResetWorker` — at daily/weekly/monthly boundaries, emits a `BUDGET_WINDOW_RESET` audit event; counter rollover is implicit via Lua window-period.
2. `L3Flusher` — every `flush_interval_seconds`, drains `PendingFlushQueue` into Prisma using upsert by `(subject, policy_id, bucket_key)`. Idempotent.
3. `AuditFlusher` — drains the audit queue into the configured sinks; runs everywhere (no leader lock), since the prisma sink uses request_id as the dedup key.

We REUSE `litellm.proxy.db.db_transaction_queue.pod_lock_manager.PodLockManager` rather than write a new lock. It already provides Redis-backed leadership, is exercised in production, and replacing it is out of scope for governor.

## Audit log

Schema (`AuditEvent`), enforced via `audit/emitter.py`:

| Field | Type | Notes |
| --- | --- | --- |
| request_id | str | the request's litellm_call_id |
| decided_at | datetime UTC | from injectable clock |
| latency_ms | float | how long admit() took |
| subjects | list[Subject] | the full hierarchy that was governed |
| decision | Status | admitted | admitted_degraded | rejected |
| verdicts | list[Verdict] | per-policy outcome with reason and limit |
| outcome | Outcome | None | populated on reconcile |
| correlation | dict[str, str] | trace_id, span_id when otelv2 active |

Transport: pluggable sinks. Default is a new Prisma table `LiteLLM_GovernanceAuditLog` with a TTL-based retention column (`expires_at`, default 30 days, configurable per-policy). Optional sinks: structured `verbose_proxy_logger` for ops who want stdout-only, and an `otel` sink that emits each audit as a structured log record via the OpenTelemetry Logs API. The OTel sink is the recommended path for regulated workloads: a Collector pipeline filters audit-tagged records into an append-only/immutable store (S3 Object Lock or equivalent) with long retention, separate from the 30-day observability pipeline. Audit content is content-hashed at the source for tamper-evidence; the Collector persistent sending queue keeps audit events durable across restarts.

Audit lives in `audit/` as a sibling, not under `engine/`, for the same reason mappers are siblings of the emitter in otelv2: the engine should not know how an audit gets shipped.

Append-only by contract; updates and deletes are explicitly not supported by the sinks. Retention is governed by a background prune in the prisma sink running every 6 hours under pod lock.

## What we replace, what we coexist with

| Legacy file | Governor relationship | Dispatch |
| --- | --- | --- |
| `litellm/proxy/spend_tracking/budget_reservation.py` | coexist via gate | `proxy_server` calls `reserve_budget_for_request` only when gate is off; when on, the `GovernorV2` callback's `async_pre_call_hook` is the budget enforcement |
| `litellm/proxy/hooks/parallel_request_limiter_v3.py` | coexist via gate | Same: the existing callback registration in `proxy_server.startup` is gated on `not is_governor_v2_enabled()`; governor registers its own callback when the gate is on |
| `litellm/proxy/hooks/max_budget_limiter.py` | coexist via gate | Same |
| `litellm/proxy/hooks/model_max_budget_limiter.py` | coexist via gate | Same |
| `litellm/proxy/hooks/dynamic_rate_limiter_v3.py` | coexist via gate | Same |
| `litellm/proxy/hooks/parallel_request_limiter.py` (older v2) | coexist; untouched | Already inactive in modern deployments |
| `estimate_request_max_cost` helper from `budget_reservation.py` | reuse | Extracted to a shared util that both legacy and governor call |
| `LiteLLM_VerificationToken.spend` column | shared | L3 flusher writes to it (idempotent upsert) so legacy reads still see correct totals |
| `pod_lock_manager` | reuse | All governor workers acquire under it |

The single dispatch site is `litellm/proxy/proxy_server.startup_event`, where a one-line `if is_governor_v2_enabled():` chooses which callback chain to register. No code in `proxy_server.py` knows about governor internals.

## Migration story

Roll forward: set `LITELLM_GOVERNOR_V2=true` in one pod, restart. The legacy callbacks no-op (registration gated), governor's callback registers. The L3 flusher upserts the same `LiteLLM_VerificationToken.spend` column legacy uses, so a mid-flight rolling restart (some pods on legacy, some on governor) still converges to the same totals; the legacy pod reads from spend and the governor pod reads from Redis, but reconcile keeps both within `flush_interval_seconds` of each other. Roll back: unset the env var and restart; legacy callbacks register, governor's counters in Redis age out via TTL within one window, and `LiteLLM_VerificationToken.spend` is already authoritative. No schema migration is reversible-blocking; the `LiteLLM_GovernanceAuditLog` table is additive and can sit dormant.

## Open questions

1. **Where does cost estimation live?** Currently in `budget_reservation.py`. Move to `litellm/litellm_core_utils/cost_estimation.py` so both legacy and governor call the same helper. DECIDED.
2. **Audit sink default.** Default to `verbose_logger`; Prisma is opt-in. Customers who need durable audit set it in config; everyone else gets stdout-shaped logs without DB write pressure. DECIDED.
3. **Tag hierarchy bound.** Cap matched tag verdicts at 10 per request; when a request carries more, the first 10 are evaluated individually and the rest are folded into one aggregate "overflow" verdict in the audit. Keeps audit row size sane. DECIDED.
4. **TPM accuracy.** Sliding-window-counter (two-bucket interpolation) is fine for TPM. The drift is fail-open by design, the unit is fungible tokens, and the alternative (sliding-window-log) is materially more expensive. DECIDED.

## Review revisions (supersedes anything conflicting above)

The adversarial review surfaced four classes of bug the original design would have shipped. These supersede earlier sections; implementers must follow these.

### R1 — Counter eviction is a TierDegraded signal, not a zero

Redis with `maxmemory-policy allkeys-lru` or `volatile-lru` can evict a hot counter mid-window while the window_key survives. The original Lua read `GET counter_key or '0'` and would silently re-admit from zero — the exact spend-counter postmortem shape, reborn.

Fixes:

- On startup, the engine queries `CONFIG GET maxmemory-policy`. If the policy is anything other than `noeviction` (or `volatile-*` with budget keys persisted via no-expiry), log CRITICAL and refuse to start when fail-closed policies are configured. Operators get a clean error, not silent leak.
- In `check_and_increment.lua`: a missing counter when the window is live signals eviction. Return a new sentinel `{ 3, "evicted_mid_window" }`; the Python wrapper raises `TierDegraded(tier="L2", reason="evicted_mid_window")`. Fail-closed policies reject; fail-open policies admit_degraded. Never reseed to zero.
- The Lua treats counter-missing as eviction ONLY when the window_key exists and the window has not elapsed. First-write-of-window is unchanged.

### R2 — Reconcile uses a request-scoped released flag, not a bloom

The original 60s bloom was wrong on a money path: false positives drop deltas (undercount), per-pod state can't dedup cross-pod success-on-A/failure-on-B (double-apply), and >60s slow streams (think long Anthropic streaming responses) double-apply.

Fix: reconcile dedups via an exact Redis key, set atomically inside the same EVAL that applies the delta:

```
SET reconciled:{request_id}:{policy_id} 1 NX EX <max_stream_lifetime>
```

If `NX` returns false, the delta is dropped. TTL defaults to 1 hour (configurable per-policy); that's the bound on stream lifetime where double-fire matters. The legacy `TPM_RESERVATION_RELEASED_KEY` metadata flag in `budget_reservation.py` is the same idea; we use Redis instead of metadata because the request_id is the durable handle.

### R3 — Budget admission always takes the L2 write path

L1 staleness is fine for read-only header hints (`X-RateLimit-Remaining`), but if budget admission reads from L1 with a 1s staleness window, two pods can each admit a 30¢ request off the same 70¢-of-$1-budget snapshot — TOCTOU, immediate over-spend.

Fix: `BudgetPolicy.admit` and `ConcurrentRequestPolicy.admit` always call `check_and_increment` (the L2 write path). The L1 cache exists for:

- Header construction after the L2 write returns the new value (zero extra latency).
- Read-only display endpoints (`/spend/...`).
- Rate-only policies where exact precision is not required (RPM/TPM fail-open).

The cache facade exposes two methods: `admit_via_l2(...)` (always L2 atomic) and `read_for_header(...)` (L1-with-staleness). Policies choose explicitly; budgets are wired to `admit_via_l2`. No more `cache.get(...)` ambiguity.

### R4 — Reconcile uses a dedicated Lua script, never window-reset semantics

The original suggested "second EVAL of `check_and_increment.lua` with `limit=-1`" to apply the actual-vs-estimated delta. If reconcile lands after a budget window has rolled (slow stream completing past midnight), the window-reset branch fires with the negative delta and seeds the new window negative.

Fix: keep `reconcile_actual.lua` as a separate script (as the file manifest already shows). It never accepts a window_period argument, never enters the window-reset branch, and only applies `INCRBYFLOAT` with the bounded delta. If the counter was already evicted (R1), the dedicated reconcile script returns `TierDegraded(reason="reconcile_against_evicted")` and the audit records the lost delta as a CRITICAL governance event.

The file manifest already lists `reconcile_actual.lua`; the prose contradiction is hereby resolved in favor of the manifest.

### R5 — Counter key shape carries Redis Cluster hash tags

`keys.counter_key(...)` and `keys.window_key(...)` MUST produce keys that hash to the same slot under Redis Cluster, or `EVAL` with two KEYS throws `CROSSSLOT`. Format:

```
spend:{<policy_id>:<subject_kind>:<subject_id>}:counter
spend:{<policy_id>:<subject_kind>:<subject_id>}:window
reconciled:{<policy_id>:<request_id>}
```

The `{...}` braces are Redis Cluster's hash tag syntax: only what's inside `{...}` is hashed, so both `:counter` and `:window` colocate. Same for reconcile dedup keys when bundled with their policy.

### R6 — Ambiguous-timeout retry is removed from write paths

The original `_eval_with_retry` retried once on timeout, but a timeout can mean "the EVAL didn't run" OR "the EVAL ran and the network ate the reply." Retrying the latter double-applies.

Fix: read-only paths may retry once. Write paths (`check_and_increment`, `reconcile_actual`, concurrent INCR/DECR) NEVER retry — a timeout raises `TierDegraded(reason="redis_timeout_ambiguous")` and the engine's fail-mode logic decides. For reconcile specifically, the NX dedup key from R2 makes a retry SAFE in principle (the second EVAL sees the dedup key and drops), but the simpler "no retry on writes" rule is easier to audit and we keep it.

### R7 — PolicyDescriptor seam, not five-touchpoint enums

Adding a new policy class should touch ONE registration point, not five (config, counter kind, reject reason, response mapping, registry). The original design wired each of those independently — worse evolvability than otelv2's mapper registry.

Fix: introduce `PolicyDescriptor`:

```python
@dataclass(frozen=True)
class PolicyDescriptor:
    name: str
    config_cls: type[BasePolicyConfig]
    policy_cls: type[Policy]
    counter_kind: Literal["spend", "rpm", "tpm", "inflight", "cost_throttle"]
    reject_code: str            # e.g. "BUDGET_EXCEEDED"
    http_status: int            # 429 or 403
    default_fail_mode: FailMode
```

Built-in descriptors live in `policies/registry.py`; the engine, adapter, and audit derive their behavior from the descriptor. Adding a new policy = one descriptor + one Policy class. That's the only seam.

### R8 — `aiolimiter` is dropped from the design

It's per-event-loop, which means per uvicorn worker per pod. With `--num_workers 4` × 10 pods, a "soft 100 RPM" cap becomes 100 × 4 × 10 = 4000 RPM. Saving one Redis RTT is not worth the inaccuracy. Use the L1 read-through cache for header hints only; never trust a per-loop limiter for enforcement.

### R9 — Typed correlation, headers, and metadata

Replace `Mapping[str, str]` bags with frozen dataclasses where possible:

```python
@dataclass(frozen=True)
class Correlation:
    trace_id: str | None
    span_id: str | None
    call_id: str

@dataclass(frozen=True)
class RateLimitHeaders:
    limit: int | None
    remaining: int | None
    reset_seconds: int | None
    reason: str | None
    degraded: bool

@dataclass(frozen=True)
class AdmitContext:
    request_id: str
    subjects: SubjectRef
    model: str
    estimated_cost: float | None
    estimated_tokens: int | None
    metadata: Mapping[str, str]   # genuinely arbitrary user metadata
```

`AdmitContext` becomes a frozen dataclass like its siblings (was TypedDict for no reason). `metadata` stays a Mapping[str, str] because it really is arbitrary, but everything that has a known shape is dataclassed.

The `Policy` Protocol takes a `CounterStore` Protocol instead of the concrete `ThreeTierCache`, so tests can inject a `FakeCounterStore` without touching tiers.

### R10 — Drops and folds

- `./governor_pending_flush.log` on-disk WAL: dropped. Ephemeral and read-only filesystems are common; an unread WAL is theatre. Postgres outage >10min logs CRITICAL and the queue overflow is recorded in audit. The L2 counters survive a pod restart anyway.
- `WindowResetWorker`: folded into the L3 flusher. Window rollover is implicit in the Lua; the flusher emits the audit breadcrumb at the boundary. No dedicated worker.
- `CostThrottlePolicy` hot-path sleeps: not in v1. v1 rejects-early at the configured threshold (e.g. reject at ≥95% of budget); probabilistic admit and graduated delays are a follow-up once we have telemetry on whether they help.

### Operational note

`pod_lock_manager` is Redis-backed. An L2 brownout simultaneously fail-closes budget policies AND drops flusher leadership. That's acceptable for v1 (the flusher catches up on the next leadership election, and the in-memory `PendingFlushQueue` survives the gap), but operators should know about it. The Open Question on asyncpg advisory locks stays a follow-up.
