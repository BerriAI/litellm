# LiteLLM client-side queuing and backpressure

## TL;DR

No generic client-side request queue. There are two capacity gates, plus one true
request-level queue that releases when a healthy deployment frees up.

| Mechanism | What it does when full | Where |
| --- | --- | --- |
| `max_parallel_requests` per-deployment slots | immediate 429 (`RateLimitType.CONCURRENT_REQUESTS`) | `router_utils/client_initalization_utils.py` |
| Priority scheduler queue | blocks in a per-model-group heap until a healthy deployment is available or the router timeout fires | `litellm/scheduler.py` |
| Proxy parallel-request limiter v3 | immediate 503 (`ServiceUnavailableError`) | `proxy/hooks/parallel_request_limiter_v3.py` |

## 1. `max_parallel_requests`: capacity cap, not a queue

`MaxParallelRequestsLimit` in `litellm/router_utils/client_initalization_utils.py`
holds one slot per in-flight request per deployment. When `in_flight >=
max_parallel_requests`, `acquire()` raises
`RateLimitError(RouterErrors.max_parallel_requests_exceeded, category=LITELLM_RATE_LIMIT,
rate_limit_type=CONCURRENT_REQUESTS)`. The caller gets a 429 immediately.
No waiting, no re-queueing. This is the default behaviour for
`acompletion` / `atext_completion` when no `priority` or
`default_priority` is set.

The limit is computed by
`calculate_max_parallel_requests(rpm, max_parallel_requests, tpm,
default_max_parallel_requests)` and cached under the key
`{model_id}_max_parallel_requests_client`. Which endpoint kinds acquire a
slot at all is controlled by the `_SLOTTED_KINDS` frozenset in
`litellm/router_rust/invoke/slots.py`.

## 2. Priority scheduler queue: the only built-in "queue and release" path

When a request carries a `priority` integer, or the router was constructed
with `default_priority`, the call goes through
`Scheduler.wait_for_turn` in `litellm/scheduler.py` instead of a direct
drive.

Flow (in `litellm/router_rust/invoke/endpoints.py`):

1. `_split_priority(model, kwargs)` resolves the effective priority via
   `resolve_request_priority(requested, default_priority, drops_params)`.
   `priority` is stripped from kwargs. If the result is `None`, the call
   drives the deployment directly (no queue).
2. Otherwise `_schedule(model, priority, factory, request)` builds a
   `healthy_deployments()` closure that calls
   `self._core.select(model, dict(request), kind="healthy")` and then
   `scheduler.wait_for_turn(request=FlowItem(priority, request_id,
   model_name), timeout=self._core.resolved.timeout,
   get_healthy_deployments=...)`.
3. On success the response headers get
   `x-litellm-request-prioritization-used: True`, which is the only marker
   that a request actually went through the queue.

### How the queue works

`Scheduler` (`litellm/scheduler.py`) keeps a per-model-group binary min-heap
of `(priority, request_id)` tuples. Lower number = higher priority
(0 is highest, 255 is lowest). The heap is backed by a `DualCache` with an
optional Redis backend, key `scheduler:queue:{model_name}`, and a 5-second
in-memory TTL when Redis is present. `polling_interval` defaults to 3 ms
(`DEFAULT_SCHEDULER_POLLING_INTERVAL` in `litellm/constants.py`).

`wait_for_turn` pushes the request onto the heap, then loops:
- If `poll(request, health_deployments)` returns True the request proceeds.
- Otherwise sleep `polling_interval` and re-check.
- When the router `timeout` is reached the request is removed from the heap
  and a `Timeout` exception is raised.

`poll` returns True when either
- there is at least one healthy deployment, or
- the request is at the top of the queue (and it is popped, so the next
  poller sees the following entry).

If a queue entry has been evicted from the cache (TTL expired or a
concurrent writer overwrote it), the request is re-enqueued at its
priority so it keeps its place in the order instead of jumping ahead or
failing.

`schedule_acompletion(model, messages, priority=None, stream=False,
**kwargs)` at `endpoints.py` L2008-2019 is the explicit queueing entry
point: it imports `priority_error` and `unqueueable_priority_message` and
raises when `priority` is `None` or not queueable.

### When a request is "unqueueable"

`unqueueable_priority_message(requested)` in
`litellm/router_utils/request_priority.py` returns:
- `"priority must be an integer, got <type> <value>, and there is no
  router_settings.default_priority to queue it at instead"` when the value
  is not a valid int and no default is configured, or
- `"priority is required to queue a request, send one or set
  router_settings.default_priority"` when no priority was supplied at all.

`resolve_request_priority` returns `None` (i.e. no queue) when the request
supplied no `priority` and the router has no `default_priority`. It
returns `default_priority` instead of the requested value when
`drops_params` is true, so a bad value is silently dropped rather than
rejected. `parse_default_priority` treats `True`/`False` as
`InvalidPriority` because a bool is not a valid int here.

### `drop_params` interaction

`request_drops_params(kwargs, router_defaults, deployment_params)`
combines three sources: the global `litellm.drop_params`, the
`additional_drop_params` lists on request / router / deployment, and the
explicit `drop_params` flag on any of those scopes. When the result is
true, a bad `priority` value is coerced to the router's `default_priority`
instead of raising, and `priority_header_fields` returns an empty mapping
so no header is emitted.

## 3. Proxy parallel-request limiter v3

`_PROXY_MaxParallelRequestsHandler_v3` in
`litellm/proxy/hooks/parallel_request_limiter_v3.py` is a Redis-backed,
sliding-window parallel-request limiter for proxy deployments. It is not a
queue: when all slots are taken the request is rejected with
`HTTP 503 Service Unavailable`
(`ServiceUnavailableError`). It also carries the priority-based rate-limit
policy: when the limiter is saturated (all slots in use), per-key
`default_priority` weights are enforced (see
`litellm/proxy/hooks/dynamic_rate_limiter_v3.py`, L127 / L223). This is a
fair-share gate, not a wait-then-release queue.

The proxy also exposes `/health/backlog` (backed by
`litellm/proxy/middleware/in_flight_requests_middleware.py`) that reports
per-pod in-flight request depth, but this is an observability endpoint
only; it does not drive any queueing.

## 4. What is NOT in the codebase

- No generic client-side request queue that blocks until capacity frees
  up. The default path (no `priority`) is fail-fast: 429 from
  `MaxParallelRequestsLimit` at the router level, 503 from the proxy v3
  limiter.
- No backpressure signalling to the caller (no `Retry-After`, no
  `X-Queue-Position`, no long-poll). The only signalling is the
  `x-litellm-request-prioritization-used` response header on requests
  that did go through the scheduler queue.
- No automatic client-side re-queueing of rejected requests. The caller
  must implement its own retry with backoff; the router's built-in
  `num_retries` / `cooldown_time` retry loop only applies within a single
  call to `acompletion` and does not re-enqueue onto the priority
  scheduler.

## 5. Summary of the "release when capacity frees up" path

The only built-in mechanism that waits and releases is the priority
scheduler queue. To use it:

1. Set `default_priority` on the router (or pass `priority` per request).
2. The request enters the per-model-group heap.
3. `wait_for_turn` polls every 3 ms. When a healthy deployment becomes
   available, or the request reaches the head of the queue, it proceeds.
4. The response carries `x-litellm-request-prioritization-used: True`.
5. The router `timeout` bounds the maximum wait; on expiry the request is
   removed from the queue and a `Timeout` is raised.

Without a priority, there is no queue. `max_parallel_requests` is a hard
cap that produces an immediate 429 when exhausted, and the proxy v3
limiter produces an immediate 503.

## 6. Solution options for client-side queuing / backpressure

The goal: when capacity is unavailable, the request is queued and
released once a slot frees up, instead of being rejected with 429/503.

### Option 1: Use the existing priority scheduler as-is

Set `default_priority` on the router, or pass `priority` per request.
The request is pushed into the per-model-group min-heap and
`wait_for_turn` polls until the request reaches the head of the queue
and a deployment is available, or the `timeout` expires.

Pros: zero new code; documented today; multi-worker safe when
Redis-backed (the `DualCache` in front of the heap); the
`x-litellm-request-prioritization-used` response header already marks
queued requests.

Cons:
- The scheduler queue is keyed per model group, not per deployment
  slot. If one of N deployments in a group has a lower
  `max_parallel_requests` than the others, the queue does not know
  which slot will free up first; it just releases in priority order.
- Priority is a relative ordering, not a capacity signal: two
  concurrent requests with the same priority both queue, but neither
  knows the other is ahead of it.
- `default_priority` on the router makes *every* request
  queueable. The user has to opt out per request by passing
  `priority=None` explicitly (or not set the default), which is a
  footgun: a request that previously fail-fast'd now silently blocks
  up to `timeout` seconds.
- `unqueueable_priority_message` (see
  `litellm/router_utils/request_priority.py`) already blocks a list of
  request kinds from the queue; streaming is *not* in that list today
  (open question: should it be?).

### Option 2: Client-side retry loop with `Retry-After` / backoff

Leave the router fail-fast (429 / 503). The client wraps the call in a
retry loop: on 429/503 it sleeps and re-sends. Optionally the server
adds a `Retry-After` header with an estimated wait so the client does
not have to guess.

Pros:
- Zero server-side queue state; the client is the queue.
- Works with any upstream that already sends 429/503, not just
  LiteLLM.
- `litellm/litellm_core_utils/get_model_cost_map.py` already has a
  `Retry-After`-aware backoff helper that the client can borrow.

Cons:
- Every rejected request is a full round-trip failure. The client has
  to deal with idempotency, timeouts, and partial state (e.g. a
  streaming response that started but the retry will re-send the whole
  request).
- Under sustained overload the client hammers the server with retries
  and amplifies load. The server still has to send the 429, so the
  admission cost is paid on every attempt.
- No ordering guarantee: requests released by the server do not
  resume in priority order, they resume in whatever order the clients
  happen to re-send.

### Option 3: Server-side queue in front of `MaxParallelRequestsLimit`

Add a bounded FIFO (or priority-aware FIFO) between the router
entrypoint and `MaxParallelRequestsLimit.acquire()`. When the slot
count is full, the request is enqueued instead of raised; a background
drain releases requests as slots free. The drain can be driven by a
`Condition` wait in the in-memory case, or by a Redis pub/sub /
`BRPOP` in the multi-worker case.

Pros:
- A true "wait for capacity" semantic, exactly what the user asked for.
- Per-deployment keying is possible: the queue knows which slot it is
  waiting on, so a release of deployment A only drains queue A.
- Works with the existing `max_parallel_requests` value, no change to
  the cap.

Cons:
- A new subsystem with two storage backends (in-memory `Condition`
  vs Redis) and a drain loop that has to release slots atomically
  with the limiter's acquire/release, which is a real concurrency
  hazard.
- Streaming requests are not queueable today (the response is already
  in flight once the first token is sent), so the queue can only
  hold non-streaming requests; that splits the API in two and the
  user has to pick.
- Bounded queue means a second failure mode: queue overflow. The
  overflow policy (fail-fast, 503 with `Retry-After`, or reject by
  priority) is a third configuration knob on top of `max_parallel_requests`
  and `timeout`.
- The proxy v3 limiter (503 gate) fires *before* the router, so the
  proxy path would need its own queue, or the gate would be bypassed
  by the new queue, which is a regression.

### Option 4 (recommended): Opt-in bounded scheduler-queue wait via a
new router setting

Reuse the existing `Scheduler` / `wait_for_turn` mechanism, but add a
new opt-in router setting, e.g. `slot_wait_timeout` (default `None`),
that, when set, routes *all* requests through the scheduler queue for
a bounded wait before falling back to fail-fast. The current
`default_priority` already does "all requests queue" but without a
separate wait cap and without making the queue a first-class
backpressure knob.

Concretely:
- New `Router(..., slot_wait_timeout=30.0)` setting, plumbed through
  `RustSettings` / `_settings_snapshot` in
  `litellm/router_rust/core.py`.
- In `_split_priority` / `_schedule` (see
  `litellm/router_rust/invoke/endpoints.py`), when
  `slot_wait_timeout` is set and no per-request `priority` is given,
  the request is enqueued with a neutral priority (e.g. the existing
  `default_priority` or a new internal sentinel) and the wait is
  bounded by `min(slot_wait_timeout, timeout)`.
- On wait expiry the request is removed from the queue and the
  existing 429 / 503 is raised, so the failure mode is unchanged.
- A new response header (e.g.
  `x-litellm-queued: True`) marks requests that went through the
  queue, so a client can distinguish "admitted immediately" from
  "waited and released".

Pros:
- Reuses the proven `Scheduler` / `wait_for_turn` path, including the
  Redis-backed `DualCache` and the per-model-group heap. No new
  storage, no new drain loop.
- Opt-in: default behavior stays fail-fast, so existing callers are
  unaffected.
- The wait is explicitly bounded and separate from the overall
  `timeout`, so a queued request can be capped at, e.g., 30 s of
  queue time even if the call's total `timeout` is 120 s.
- Per-deployment keying can come later: the scheduler queue is per
  model group today, which is a coarser key than the slot, but it is
  the same key the user already configures via `model_group` /
  `deployment` lists, so it is not a new concept.
- The new response header gives the client a real signal for
  observability ("this request waited N s in queue"), which the
  existing `x-litellm-request-prioritization-used` header already
  provides for the priority path.

Cons:
- Per-model-group keying means the queue cannot distinguish "slot A
  of 3 free" from "slot B of 3 full"; the queue releases in priority
  order, not in slot-free order. This is the same limitation as
  option 1 and is acceptable for a first iteration.
- The proxy v3 503 gate still fires before the router queue, so a
  request that would have queued still gets a 503 from the proxy
  when the proxy-level gate is saturated. That is a known limitation;
  the gate and the queue have different semantics (gate = "too many
  in flight right now", queue = "wait your turn") and the user has to
  configure the gate's limit above the queue's expected depth.
- A new router setting means a new config surface to document and to
  keep in sync with the Rust-side settings snapshot.

### Recommendation

Ship option 4 first. It is the smallest change that gives the user
the "queue and release" semantic they asked for, reuses the existing
scheduler queue, and keeps the default fail-fast behavior intact. The
new `slot_wait_timeout` setting and the `x-litellm-queued` header are
the two new surface areas; everything else is plumbing the request
through a path that already exists.

The two limitations of option 4 most likely to block adoption have
explicit remediations, below. Both are independent of the option 4
plumbing and can land as follow-up PRs.

Follow-ups to consider after option 4 lands:
- Add `Retry-After` to the 429 / 503 responses so that a client that
  does *not* opt into the queue can still back off sensibly
  (overlaps with option 2).

#### Remediation 1: proxy v3 503 gate bypasses the router queue

Root cause. `_PROXY_MaxParallelRequestsHandler_v3.async_pre_call_hook`
(`litellm/proxy/hooks/parallel_request_limiter_v3.py`, `async_pre_call_hook`
at the v3 handler class) runs at proxy entry, before the router. Its
`should_rate_limit` reads the in-flight `parallel_count` gauge and, when
the gauge is at or above `max_parallel_requests`, returns `OVER_LIMIT`
and `_handle_rate_limit_error` raises a 503. The router queue never
gets to see the request, so a saturated proxy gate defeats the queue
even when the router would have admitted the request.

Design decision: gate the gate by queue opt-in. The proxy v3 limiter
does not have a per-model-group queue knob. The router does (the
`slot_wait_timeout` opt-in in option 4). The gate and the queue are
coupled only when the queue is on; when the queue is off, the gate
keeps its current fail-fast 503.

Two variants, ordered by blast radius:

A. Deferred enforcement at queued admission (recommended). When the
router is running with `slot_wait_timeout` set on a model group, the
proxy v3 gate stops enforcing `max_parallel_requests` for that model
group at pre-call and instead records a *pending* slot reference on the
request stash. The router's queued admission path (where the scheduler
pops the request off the heap and a deployment becomes healthy)
re-checks the slot count at that moment and raises the existing 429/503
if the slot is actually taken. This keeps the gate's count semantics
(still 429/503 on real capacity exhaustion) but moves the *timing* of
the check to the moment when a slot actually frees, so a request that
is going to queue does not get a 503 just because the gauge is at its
peak.

Mechanics:
- `parallel_request_limiter_v3.py` gets a new knob
  `defer_parallel_enforcement: str | None` (a model-group allowlist).
  When the request's `model_group` is in the list, `should_rate_limit`
  skips the parallel-count branch and stashes a `pending_parallel_slot`
  marker on the `ParallelSlotAcquisition`.
- A new `DeferredParallelSlotCheck` runs on the router side immediately
  after `wait_for_turn` returns True and before `run()`. It calls back
  into the proxy's `parallel_count` gauge and raises the same
  `ServiceUnavailableError` if the slot is taken.
- The 429/503 the client sees on a deferred miss is identical to
  today's, but it fires only when a slot is actually unavailable at
  the moment of admission, not at pre-call peak.

B. Soft gate with a higher limit (operational stopgap). If the
deferred check is too large a change to ship in the same PR, configure
`max_parallel_requests` on the proxy gate to a value *above* the
expected queue depth for that model group, and document that the gate
becomes advisory for queued traffic. This is not a code change; it is
an operational knob. It only works if the operator can size the
gate's limit to the expected queue depth, which is not always known.
Document B while A is in flight.

Tests (regression against the original defect):
- Unit: a fake proxy limiter returns `OVER_LIMIT` on pre-call when the
  gate is hard; the same limiter, with `defer_parallel_enforcement`
  covering the model group, returns `OK` on pre-call and raises only
  when the deferred check fires.
- Integration: two requests race into a group whose gate limit is 1
  and whose queue limit is 2. Under variant A the second request queues
  and completes after the first releases; under variant B it 503s.
  Assert the queued request completes with `x-litellm-queued: True`
  set, and that the deferred miss path returns 429/503 with no
  `x-litellm-queued` header.

#### Remediation 2: queue keyed per model group, not per deployment slot

Root cause. `Scheduler.get_queue` / `save_queue` use
`scheduler:queue:{model_name}` as the cache key
(`litellm/scheduler.py`, `get_queue` and `save_queue`). All
deployments in a group share one heap. `poll` returns True as soon as
*any* deployment in the group is healthy, and pops in priority order,
not in slot-free order. So if deployment A has 2 slots and
deployment B has 10, the queue treats them as one shared pool of 12
and releases requests that are ahead in priority even if their actual
target deployment is still full.

Design decision: re-key the heap on `(model_name, deployment_id)` and
make `poll` slot-aware. Three coupled changes:

A. Per-deployment queue key.
- `FlowItem` (`litellm/scheduler.py`) gains `deployment_id: str |
  None` (default `None` for the legacy group-keyed path, so the
  existing `x-litellm-request-prioritization-used` header contract is
  unchanged for callers that do not opt in).
- `Scheduler.get_queue` / `save_queue` / `remove_request` take a
  `cache_key` parameter instead of deriving `model_name` alone. The
  key is built as `scheduler:queue:{model_name}:{deployment_id or
  "_"}`. The `SchedulerCacheKeys.queue` enum value is reused as the
  prefix; the suffix is the deployment id.
- `Scheduler.add_request` / `wait_for_turn` forward the deployment id
  from the `FlowItem`. The in-memory `self.queue` (the no-Redis fast
  path) stays a single list for legacy callers; when `deployment_id`
  is set, `Scheduler` holds a `dict[str, list[QueueEntry]]` keyed by
  deployment id.
- The router call site (`litellm/router_rust/invoke/endpoints.py`,
  `_schedule`) builds the `FlowItem` with `deployment_id=None` today.
  The new call site for the per-deployment path resolves a deployment
  from `select(model, dict(request), kind="healthy")` *before*
  `wait_for_turn`, and threads that deployment id into the `FlowItem`.
  If `select` returns empty, the request waits on the group-keyed
  queue (legacy) rather than on a specific deployment.

B. Slot-aware poll.
- The current `poll` (`litellm/scheduler.py`) returns True when
  `len(health_deployments) > 0` *or* the entry is at the head of the
  queue. For the per-deployment path, the head-of-queue branch is
  wrong: a request at the head of the *group* heap does not know which
  of N deployments will free a slot next. Replace it with: for the
  entry's target deployment, ask the router's
  `MaxParallelRequestsLimit` for the deployment's current `in_flight`
  and `max_parallel_requests`. If `in_flight < max`, the deployment
  can accept; pop the entry from that deployment's per-deployment heap
  and proceed. If `in_flight == max`, the entry stays in the heap and
  the next poller on that deployment id is the one who releases when
  a slot actually frees.
- The group-keyed legacy path (`deployment_id=None`) keeps today's
  `poll` semantics, so the priority-scheduler feature today is not
  broken.
- The release side: when `MaxParallelRequestsLimit.release`
  decrements `in_flight` for a deployment, it publishes a `slot_freed`
  event on a Redis pub/sub channel `litellm:slots:{deployment_id}`.
  The per-deployment poller in `wait_for_turn` subscribes to that
  channel and wakes early on the event instead of polling every 3 ms.
  In the single-worker in-memory case, the pub/sub channel is replaced
  by an `asyncio.Condition` per deployment id.

C. Header contract.
- The response header `x-litellm-queued: True` is set on the success
  path for either queue keying, so a client that opts into the queue
  gets the same marker regardless of which keying it used.
- A new header `x-litellm-queued-key: <model_name>[:<deployment_id>]`
  is set so an operator can audit which keying a request used. This is
  additive and does not break the existing
  `x-litellm-request-prioritization-used` contract.

Tests (regression against the original defect):
- Unit: `Scheduler` with a `deployment_id`-keyed `FlowItem` places the
  entry in a separate heap from another deployment in the same group;
  a release on deployment A wakes only deployment A's poller.
- Unit: `poll` for the per-deployment path returns False when the
  target deployment's `in_flight` is at its `max_parallel_requests`,
  and True the moment `in_flight` drops below it, regardless of where
  the entry sits in the group heap.
- Integration: a group with deployment A (`max_parallel_requests=1`)
  and B (`max_parallel_requests=10`), 12 concurrent requests all
  targeting the group, no explicit priority. Under the group-keyed
  legacy path, the first 12 to pop in priority order complete (B
  absorbs the load). Under the per-deployment path with
  `deployment_id` set, the requests that resolve to A queue behind A's
  single slot; the requests that resolve to B complete as B's 10 slots
  open. Assert `x-litellm-queued-key` reflects the deployment id.
- Backward compat: the legacy `x-litellm-request-prioritization-used`
  header is still set on the group-keyed path, and no new header
  appears when the caller does not opt into per-deployment keying.

Sequencing. The two remediations are independent and can ship as
separate PRs. Remediation 2 is the bigger change (heap re-key,
pub/sub channel, router call-site plumbing) and should land first so
that the deferred gate check in remediation 1 has a stable
per-deployment slot count to consult. If both land in one release,
the per-deployment keying should be opt-in (`slot_wait_timeout`
carries a `queue_keying: "group" | "deployment"` field, default
`group`) so existing priority-scheduler users are not migrated
automatically.
