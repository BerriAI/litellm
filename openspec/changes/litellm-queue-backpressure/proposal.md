# Change: LiteLLM client-side queuing and backpressure

## Why

LiteLLM has no generic client-side request queue. When capacity is
exhausted the default path is fail-fast: `max_parallel_requests` raises a
429 immediately at the router, and the proxy parallel-request limiter v3
raises a 503 at proxy entry. The only built-in "queue and release when
capacity frees" path is the priority scheduler, which is keyed per model
group rather than per deployment slot, and is only reachable when a
request carries a `priority`. The two limitations most likely to block
adoption of the recommended bounded scheduler-queue wait (option 4,
`slot_wait_timeout`) have explicit remediations:

1. The proxy v3 503 gate fires before the router queue, so a saturated
   proxy gate defeats the queue even when the router would have
   admitted the request.
2. The scheduler queue is keyed per model group, not per deployment
   slot, so it cannot distinguish "slot A of 3 free" from "slot B of 3
   full" and releases in priority order rather than slot-free order.

Source: `litellm_backpressure_analysis.md`, sections 6.4 ("Option 4
recommended") and 6.5 ("Remediation 1" / "Remediation 2").

## What Changes

Two independent capabilities, sequenced so that remediation 2 (per
deployment queue keying) lands first and remediation 1 (gate deferral)
can consult per-deployment slot counts.

### Capability: `scheduler-per-deployment-keying`

Re-key the priority scheduler queue on `(model_name, deployment_id)` and
make `poll` slot-aware, so a request waits on the specific deployment
slot it will actually occupy rather than on the whole model group.

- `FlowItem` gains `deployment_id: str | None` (default `None` keeps the
  legacy group-keyed path and its `x-litellm-request-prioritization-used`
  header contract unchanged).
- `Scheduler.get_queue` / `save_queue` / `remove_request` take a
  `cache_key` parameter. The Redis key becomes
  `scheduler:queue:{model_name}:{deployment_id or "_"}`. The in-memory
  fast path holds a `dict[str, list[QueueEntry]]` keyed by deployment id
  alongside the legacy single list.
- The router call site in `litellm/router_rust/invoke/endpoints.py`
  (`_schedule`) resolves a healthy deployment via
  `select(model, dict(request), kind="healthy")` *before*
  `wait_for_turn` and threads its id into the `FlowItem`. If `select`
  returns empty, the request falls back to the group-keyed queue.
- `poll` for the per-deployment path asks the router's
  `MaxParallelRequestsLimit` for the target deployment's `in_flight`
  and `max_parallel_requests`. It returns True when `in_flight <
  max` (pop and proceed) and False when `in_flight == max` (stay in
  that deployment's heap). The group-keyed legacy path keeps today's
  `poll` semantics.
- On release, `MaxParallelRequestsLimit.release` publishes a
  `slot_freed` event on a Redis pub/sub channel
  `litellm:slots:{deployment_id}`. The per-deployment poller
  subscribes and wakes early instead of polling every 3 ms. In the
  single-worker in-memory case the channel is replaced by an
  `asyncio.Condition` per deployment id.
- Two response headers:
  - `x-litellm-queued: True` set on the success path for either queue
    keying.
  - `x-litellm-queued-key: <model_name>[:<deployment_id>]` set so an
    operator can audit which keying a request used.

### Capability: `proxy-gate-deferral`

Gate the proxy v3 503 gate by queue opt-in. When the router is running
with `slot_wait_timeout` set on a model group, the proxy v3 gate stops
enforcing `max_parallel_requests` at pre-call for that model group and
instead defers the slot check to the moment the router actually
admits the request, so a request that is going to queue does not get a
503 just because the gauge is at its peak.

- `parallel_request_limiter_v3.py` gains a new knob
  `defer_parallel_enforcement: str | None` (a model-group allowlist).
  When the request's `model_group` is in the list, `should_rate_limit`
  skips the parallel-count branch and stashes a `pending_parallel_slot`
  marker on the `ParallelSlotAcquisition`.
- A new `DeferredParallelSlotCheck` runs on the router side immediately
  after `wait_for_turn` returns True and before `run()`. It calls back
  into the proxy's `parallel_count` gauge and raises the same
  `ServiceUnavailableError` if the slot is taken.
- The 429/503 the client sees on a deferred miss is identical to
  today's, but it fires only when a slot is actually unavailable at the
  moment of admission, not at pre-call peak.
- An operational stopgap (variant B, documented while the code change
  is in flight): configure the proxy gate's `max_parallel_requests`
  above the expected queue depth for that model group and document that
  the gate becomes advisory for queued traffic.

## Sequencing

The two capabilities are independent and can ship as separate PRs.
`scheduler-per-deployment-keying` is the bigger change (heap re-key,
pub/sub channel, router call-site plumbing) and should land first so
that the deferred gate check in `proxy-gate-deferral` has a stable
per-deployment slot count to consult. If both land in one release, the
per-deployment keying should be opt-in (`slot_wait_timeout` carries a
`queue_keying: "group" | "deployment"` field, default `group`) so
existing priority-scheduler users are not migrated automatically.

## Impact

Affected specs: `scheduler-per-deployment-keying`, `proxy-gate-deferral`
(new capabilities).

Affected code:
- `litellm/scheduler.py` (`FlowItem`, `Scheduler.get_queue` /
  `save_queue` / `remove_request` / `add_request` / `wait_for_turn` /
  `poll`)
- `litellm/router_utils/client_initalization_utils.py`
  (`MaxParallelRequestsLimit.release` publishes `slot_freed`)
- `litellm/router_rust/invoke/endpoints.py` (`_schedule` resolves the
  deployment before `wait_for_turn`; new `DeferredParallelSlotCheck`)
- `litellm/router_rust/invoke/slots.py` (`_SLOTTED_KINDS`)
- `litellm/proxy/hooks/parallel_request_limiter_v3.py`
  (`defer_parallel_enforcement` knob)
- `litellm/proxy/hooks/dynamic_rate_limiter_v3.py` (priority-based
  rate-limit policy when the limiter is saturated)

New response headers: `x-litellm-queued`, `x-litellm-queued-key`.
New Redis pub/sub channel: `litellm:slots:{deployment_id}`.
