# Spec deltas: LiteLLM client-side queuing and backpressure

## ADDED: `scheduler-per-deployment-keying`

### Requirement: `FlowItem` carries an optional `deployment_id`

`FlowItem` (`litellm/scheduler.py`) gains a `deployment_id: str | None`
field, default `None`. When `None`, the request uses the legacy
group-keyed queue and the existing
`x-litellm-request-prioritization-used` header contract is unchanged.
When set, the request is queued on the specific deployment slot it
will occupy.

#### Scenario: legacy caller does not set `deployment_id`

- **WHEN** a request is enqueued with `FlowItem.deployment_id=None`
- **THEN** the entry is placed in the legacy single in-memory queue
  list and persisted to the legacy `scheduler:queue:{model_name}` key
- **AND** the legacy `x-litellm-request-prioritization-used` header
  is still set on the priority path

#### Scenario: opt-in caller sets `deployment_id`

- **WHEN** a request is enqueued with `FlowItem.deployment_id` set
- **THEN** the entry is placed in a per-deployment heap keyed by that
  deployment id
- **AND** the response sets `x-litellm-queued: True` and
  `x-litellm-queued-key: {model_name}:{deployment_id}`

### Requirement: `Scheduler` takes a `cache_key` parameter instead of deriving `model_name` alone

`Scheduler.get_queue`, `Scheduler.save_queue`, and
`Scheduler.remove_request` take a `cache_key` parameter. The key is
built as `scheduler:queue:{model_name}:{deployment_id or "_"}`. The
`SchedulerCacheKeys.queue` enum value is reused as the prefix; the
suffix is the deployment id. `Scheduler.add_request` and
`Scheduler.wait_for_turn` forward the deployment id from the
`FlowItem`.

#### Scenario: legacy group-keyed path

- **WHEN** `deployment_id` is `None`
- **THEN** the cache key is `scheduler:queue:{model_name}:_`
- **AND** the in-memory fast path uses the legacy single list

#### Scenario: per-deployment path

- **WHEN** `deployment_id` is set
- **THEN** the cache key is `scheduler:queue:{model_name}:{deployment_id}`
- **AND** the in-memory fast path uses a `dict[str, list[QueueEntry]]`
  keyed by deployment id

### Requirement: The router resolves a healthy deployment before `wait_for_turn`

The router call site in `litellm/router_rust/invoke/endpoints.py`
(`_schedule`) resolves a healthy deployment via
`select(model, dict(request), kind="healthy")` *before*
`wait_for_turn` and threads that deployment id into the `FlowItem`.
If `select` returns empty, the request waits on the group-keyed queue
(legacy) rather than on a specific deployment.

#### Scenario: a healthy deployment exists

- **WHEN** `select(model, dict(request), kind="healthy")` returns a
  deployment
- **THEN** the `FlowItem` is built with that deployment's `deployment_id`
- **AND** the request queues on that deployment's per-deployment heap

#### Scenario: no healthy deployment exists

- **WHEN** `select(model, dict(request), kind="healthy")` returns
  empty
- **THEN** the `FlowItem` is built with `deployment_id=None`
- **AND** the request waits on the group-keyed queue

### Requirement: `poll` is slot-aware on the per-deployment path

For the per-deployment path, `poll` asks the router's
`MaxParallelRequestsLimit` for the target deployment's current
`in_flight` and `max_parallel_requests`. If `in_flight <
max_parallel_requests`, the deployment can accept: the entry is popped
from that deployment's heap and the request proceeds. If
`in_flight == max_parallel_requests`, the entry stays in the heap and
the next poller on that deployment id is the one who releases when a
slot actually frees. The group-keyed legacy path
(`deployment_id=None`) keeps today's `poll` semantics.

#### Scenario: target deployment has a free slot

- **WHEN** a per-deployment entry is polled and the target deployment's
  `in_flight` is below `max_parallel_requests`
- **THEN** `poll` returns True, the entry is popped from that
  deployment's heap, and the request proceeds

#### Scenario: target deployment is full

- **WHEN** a per-deployment entry is polled and the target deployment's
  `in_flight` equals `max_parallel_requests`
- **THEN** `poll` returns False and the entry stays in that
  deployment's heap

#### Scenario: legacy path keeps today's `poll` semantics

- **WHEN** `deployment_id` is `None`
- **THEN** `poll` returns True when `len(health_deployments) > 0` or
  the entry is at the head of the queue, matching today's behavior

### Requirement: `MaxParallelRequestsLimit.release` publishes a `slot_freed` event

When `MaxParallelRequestsLimit.release` decrements `in_flight` for a
deployment, it publishes a `slot_freed` event on a Redis pub/sub
channel `litellm:slots:{deployment_id}`. The per-deployment poller in
`wait_for_turn` subscribes to that channel and wakes early on the
event instead of polling every 3 ms. In the single-worker in-memory
case, the pub/sub channel is replaced by an `asyncio.Condition` per
deployment id.

#### Scenario: multi-worker with shared Redis

- **WHEN** `MaxParallelRequestsLimit.release` decrements `in_flight`
  for a deployment in a multi-worker deployment with a shared Redis
  instance
- **THEN** a `slot_freed` event is published on
  `litellm:slots:{deployment_id}`
- **AND** the per-deployment poller on that deployment wakes early
  instead of waiting for the next 3 ms poll

#### Scenario: single-worker in-memory

- **WHEN** `MaxParallelRequestsLimit.release` decrements `in_flight`
  for a deployment in a single-worker in-memory deployment
- **THEN** an `asyncio.Condition` per deployment id is notified
- **AND** the per-deployment poller wakes early on the condition

### Requirement: `x-litellm-queued` and `x-litellm-queued-key` headers

The response header `x-litellm-queued: True` is set on the success
path for either queue keying. A new header
`x-litellm-queued-key: <model_name>[:<deployment_id>]` is set so an
operator can audit which keying a request used. This is additive and
does not break the existing `x-litellm-request-prioritization-used`
contract.

#### Scenario: group-keyed success path

- **WHEN** a request completes via the group-keyed queue
- **THEN** `x-litellm-queued: True` is set
- **AND** `x-litellm-queued-key: {model_name}:_` is set

#### Scenario: per-deployment success path

- **WHEN** a request completes via the per-deployment queue
- **THEN** `x-litellm-queued: True` is set
- **AND** `x-litellm-queued-key: {model_name}:{deployment_id}` is set

## ADDED: `proxy-gate-deferral`

### Requirement: `defer_parallel_enforcement` is a model-group allowlist

`parallel_request_limiter_v3.py` gains a new knob
`defer_parallel_enforcement: str | None` (a model-group allowlist).
When the request's `model_group` is in the list,
`should_rate_limit` skips the parallel-count branch and stashes a
`pending_parallel_slot` marker on the `ParallelSlotAcquisition`. When
the model group is not in the list, the gate keeps its current
fail-fast 503.

#### Scenario: model group in the allowlist

- **WHEN** a request arrives at the proxy v3 gate and its
  `model_group` is in `defer_parallel_enforcement`
- **THEN** `should_rate_limit` skips the parallel-count branch
- **AND** a `pending_parallel_slot` marker is stashed on the
  `ParallelSlotAcquisition`

#### Scenario: model group not in the allowlist

- **WHEN** a request arrives at the proxy v3 gate and its
  `model_group` is not in `defer_parallel_enforcement`
- **THEN** `should_rate_limit` enforces the parallel-count branch as
  today
- **AND** a 503 is raised when the gauge is at or above
  `max_parallel_requests`

### Requirement: `DeferredParallelSlotCheck` re-checks the slot at admission

A new `DeferredParallelSlotCheck` runs on the router side immediately
after `wait_for_turn` returns True and before `run()`. It calls back
into the proxy's `parallel_count` gauge and raises the same
`ServiceUnavailableError` if the slot is taken. The 429/503 the
client sees on a deferred miss is identical to today's, but it fires
only when a slot is actually unavailable at the moment of admission,
not at pre-call peak.

#### Scenario: slot is free at admission

- **WHEN** `DeferredParallelSlotCheck` runs after `wait_for_turn`
  returns True and the target deployment's slot is free
- **THEN** the request proceeds to `run()`
- **AND** `x-litellm-queued: True` is set on the response

#### Scenario: slot is taken at admission

- **WHEN** `DeferredParallelSlotCheck` runs after `wait_for_turn`
  returns True and the target deployment's slot is taken
- **THEN** a `ServiceUnavailableError` is raised with the same 429/503
  body as today
- **AND** no `x-litellm-queued` header is set on the error response

## ADDED: regression tests

### Requirement: regression tests for the original defect

The following tests must fail before the remediations are in place and
pass once they land.

#### Scenario: proxy gate deferral, unit

- **WHEN** a fake proxy limiter returns `OVER_LIMIT` on pre-call when
  the gate is hard
- **AND** the same limiter, with `defer_parallel_enforcement` covering
  the model group, returns `OK` on pre-call and raises only when the
  deferred check fires
- **THEN** the test asserts the deferral path

#### Scenario: proxy gate deferral, integration

- **WHEN** two requests race into a group whose gate limit is 1 and
  whose queue limit is 2
- **THEN** under variant A the second request queues and completes
  after the first releases, with `x-litellm-queued: True` set
- **AND** under variant B it 503s
- **AND** the deferred miss path returns 429/503 with no
  `x-litellm-queued` header

#### Scenario: per-deployment keying, unit (heap separation)

- **WHEN** a `Scheduler` with a `deployment_id`-keyed `FlowItem` places
  the entry in a separate heap from another deployment in the same
  group
- **AND** a release on deployment A wakes only deployment A's poller
- **THEN** the test asserts heap separation and targeted wakeup

#### Scenario: per-deployment keying, unit (slot-aware poll)

- **WHEN** `poll` for the per-deployment path is called while the
  target deployment's `in_flight` is at its `max_parallel_requests`
- **THEN** it returns False
- **AND** the moment `in_flight` drops below `max_parallel_requests`
  it returns True, regardless of where the entry sits in the group
  heap

#### Scenario: per-deployment keying, integration (mixed saturation)

- **WHEN** a group has deployment A (`max_parallel_requests=1`) and
  B (`max_parallel_requests=10`), 12 concurrent requests all target
  the group, and no explicit priority is set
- **THEN** under the group-keyed legacy path the first 12 to pop in
  priority order complete (B absorbs the load)
- **AND** under the per-deployment path with `deployment_id` set, the
  requests that resolve to A queue behind A's single slot while the
  requests that resolve to B complete as B's 10 slots open
- **AND** `x-litellm-queued-key` reflects the deployment id

#### Scenario: backward compatibility

- **WHEN** the legacy `x-litellm-request-prioritization-used` header
  is still set on the group-keyed path
- **AND** no new header is set unless the caller opts in
- **THEN** existing priority-scheduler callers are not migrated
  automatically
