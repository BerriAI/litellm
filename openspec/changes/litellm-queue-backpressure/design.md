# Design: LiteLLM client-side queuing and backpressure

Source analysis: `litellm_backpressure_analysis.md`.

## Context

The recommended option 4 (a new `slot_wait_timeout` router setting that
routes all requests through the scheduler queue for a bounded wait
before falling back to fail-fast) is the smallest change that gives the
user the "queue and release" semantic while keeping default fail-fast
behavior intact. Its two limitations most likely to block adoption
have explicit remediations, which this change ships:

- **Remediation 1 (proxy gate bypass)**: the proxy v3 503 gate fires
  at proxy entry, before the router queue. A saturated proxy gate
  defeats the queue even when the router would have admitted the
  request. Fix: gate the gate by queue opt-in (deferred
  enforcement at queued admission, or a documented operational
  stopgap).
- **Remediation 2 (per-deployment keying)**: the scheduler queue is
  keyed per model group, not per deployment slot. `poll` returns True
  as soon as *any* deployment in the group is healthy and pops in
  priority order, not slot-free order. So if deployment A has 2
  slots and deployment B has 10, the queue treats them as one shared
  pool of 12 and releases requests that are ahead in priority even
  if their actual target deployment is still full. Fix: re-key the
  heap on `(model_name, deployment_id)` and make `poll` slot-aware.

## Decisions

### D1: Gate the gate by queue opt-in, not by a new per-group queue knob

The proxy v3 limiter does not have a per-model-group queue knob. The
router does (`slot_wait_timeout`). The gate and the queue are coupled
only when the queue is on. When the queue is off, the gate keeps its
current fail-fast 503. This avoids adding a second knob to the proxy
gate that would need to stay in sync with the router setting.

Two variants, ordered by blast radius:

- **A. Deferred enforcement at queued admission (recommended).**
  When the router is running with `slot_wait_timeout` set on a model
  group, the proxy v3 gate stops enforcing `max_parallel_requests` for
  that model group at pre-call and instead records a *pending* slot
  reference on the request stash. The router's queued admission path
  (where the scheduler pops the request off the heap and a deployment
  becomes healthy) re-checks the slot count at that moment and raises
  the existing 429/503 if the slot is actually taken. The gate's count
  semantics are unchanged (still 429/503 on real capacity exhaustion);
  only the *timing* of the check moves to the moment a slot actually
  frees.
- **B. Soft gate with a higher limit (operational stopgap).** If the
  deferred check is too large a change to ship in the same PR,
  configure `max_parallel_requests` on the proxy gate to a value above
  the expected queue depth for that model group and document that the
  gate becomes advisory for queued traffic. Not a code change; an
  operational knob. Only works if the operator can size the gate's
  limit to the expected queue depth, which is not always known.

Document B while A is in flight.

### D2: Re-key the heap on `(model_name, deployment_id)`; make `poll` slot-aware

Three coupled changes:

- **A. Per-deployment queue key.** `FlowItem` gains
  `deployment_id: str | None` (default `None` for the legacy
  group-keyed path, so the existing
  `x-litellm-request-prioritization-used` header contract is
  unchanged for callers that do not opt in). `Scheduler.get_queue` /
  `save_queue` / `remove_request` take a `cache_key` parameter instead
  of deriving `model_name` alone. The key is built as
  `scheduler:queue:{model_name}:{deployment_id or "_"}`. The
  `SchedulerCacheKeys.queue` enum value is reused as the prefix; the
  suffix is the deployment id. `Scheduler.add_request` /
  `wait_for_turn` forward the deployment id from the `FlowItem`. The
  in-memory `self.queue` (the no-Redis fast path) stays a single list
  for legacy callers; when `deployment_id` is set, `Scheduler` holds
  a `dict[str, list[QueueEntry]]` keyed by deployment id. The router
  call site (`_schedule`) builds the `FlowItem` with
  `deployment_id=None` today. The new call site for the
  per-deployment path resolves a deployment from
  `select(model, dict(request), kind="healthy")` *before*
  `wait_for_turn` and threads that deployment id into the `FlowItem`.
  If `select` returns empty, the request waits on the group-keyed
  queue (legacy) rather than on a specific deployment.
- **B. Slot-aware poll.** The current `poll` returns True when
  `len(health_deployments) > 0` *or* the entry is at the head of the
  queue. For the per-deployment path, the head-of-queue branch is
  wrong: a request at the head of the *group* heap does not know which
  of N deployments will free a slot next. Replace it with: for the
  entry's target deployment, ask the router's
  `MaxParallelRequestsLimit` for the deployment's current `in_flight`
  and `max_parallel_requests`. If `in_flight < max`, the deployment can
  accept; pop the entry from that deployment's per-deployment heap and
  proceed. If `in_flight == max`, the entry stays in the heap and the
  next poller on that deployment id is the one who releases when a
  slot actually frees. The group-keyed legacy path
  (`deployment_id=None`) keeps today's `poll` semantics, so the
  priority-scheduler feature today is not broken. The release side:
  when `MaxParallelRequestsLimit.release` decrements `in_flight` for a
  deployment, it publishes a `slot_freed` event on a Redis pub/sub
  channel `litellm:slots:{deployment_id}`. The per-deployment poller
  in `wait_for_turn` subscribes to that channel and wakes early on the
  event instead of polling every 3 ms. In the single-worker in-memory
  case, the pub/sub channel is replaced by an `asyncio.Condition` per
  deployment id.
- **C. Header contract.** `x-litellm-queued: True` is set on the
  success path for either queue keying, so a client that opts into the
  queue gets the same marker regardless of which keying it used. A
  new header `x-litellm-queued-key: <model_name>[:<deployment_id>]`
  is set so an operator can audit which keying a request used. This is
  additive and does not break the existing
  `x-litellm-request-prioritization-used` contract.

## Alternatives considered

- **Option 1 (use the existing priority scheduler as-is).** Zero new
  code, but the scheduler queue is keyed per model group, not per
  deployment slot, and `default_priority` on the router makes *every*
  request queueable, which is a footgun: a request that previously
  fail-fast'd now silently blocks up to `timeout` seconds.
- **Option 2 (client-side retry loop with `Retry-After` / backoff).**
  Zero server-side queue state, but every rejected request is a full
  round-trip failure, the client has to deal with idempotency,
  timeouts, and partial state, and under sustained overload the client
  hammers the server with retries and amplifies load.
- **Option 3 (server-side queue in front of `MaxParallelRequestsLimit`).**
  A true "wait for capacity" semantic with per-deployment keying, but
  a new subsystem with two storage backends and a drain loop that has
  to release slots atomically with the limiter's acquire/release,
  which is a real concurrency hazard. Streaming requests are not
  queueable today, so the queue can only hold non-streaming
  requests, splitting the API in two.

## Open questions

- Should streaming requests be added to the `unqueueable` list?
  Today streaming is *not* in that list (see
  `litellm/router_utils/request_priority.py`). Open question from the
  source analysis.
- `drop_params` interaction: when `drop_params` is true, a bad
  `priority` value is coerced to the router's `default_priority`
  instead of raising, and `priority_header_fields` returns an empty
  mapping so no header is emitted. Confirm whether the new
  `x-litellm-queued` / `x-litellm-queued-key` headers should follow
  the same `drop_params` behavior or always be emitted when the queue
  is used.

## Risks

- Re-keying the heap changes the Redis cache key for the priority
  scheduler. Existing deployments that rely on the legacy
  group-keyed path must keep working; the `deployment_id=None`
  default is the compatibility guarantee.
- The Redis pub/sub channel `litellm:slots:{deployment_id}` is a new
  channel. Multi-worker deployments that share a Redis instance will
  see it; single-worker in-memory deployments use
  `asyncio.Condition` instead and never touch the channel.
- The `DeferredParallelSlotCheck` calls back into the proxy's
  `parallel_count` gauge from the router side. The call boundary
  (router -> proxy) is new and needs to be exercised in integration
  tests with a two-worker proxy and a shared Redis instance, per the
  QA runbook in `AGENTS.md`.
