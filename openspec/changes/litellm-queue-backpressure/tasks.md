# Tasks: LiteLLM client-side queuing and backpressure

Sequencing: remediation 2 (per-deployment queue keying) lands first
because it makes the scheduler queue release in slot-free order, which
remediation 1 (proxy gate deferral) depends on for its "wait then
release" promise to hold. The two are coupled: a gate that defers
enforcement to the queue only works if the queue releases requests in
the right order.

Conventions: conventional commits, branch prefix `litellm_`, target the
default branch, regression tests that actually catch the original
defect, no AI-slop comments.

## Remediation 2: per-deployment queue keying

### `litellm/scheduler.py`

- [ ] 1.1 Add `deployment_id: str | None` field to `FlowItem`,
      default `None`.
- [ ] 1.2 Thread `deployment_id` through `Scheduler.add_request` and
      `Scheduler.wait_for_turn` into the new `cache_key` parameter.
- [ ] 1.3 Change `Scheduler.get_queue`, `Scheduler.save_queue`, and
      `Scheduler.remove_request` to take a `cache_key` parameter
      instead of deriving `model_name` alone. Build the key as
      `scheduler:queue:{model_name}:{deployment_id or "_"}`. Reuse the
      `SchedulerCacheKeys.queue` enum value as the prefix.
- [ ] 1.4 Keep the legacy in-memory single list for
      `deployment_id=None`. Add a `dict[str, list[QueueEntry]]` keyed
      by deployment id for the per-deployment path.
- [ ] 1.5 Make `poll` slot-aware on the per-deployment path: ask the
      router's `MaxParallelRequestsLimit` for the target deployment's
      current `in_flight` and `max_parallel_requests`. If
      `in_flight < max_parallel_requests`, pop the entry from that
      deployment's heap and return True. If `in_flight ==
      max_parallel_requests`, the entry stays and the next poller on
      that deployment id is the one who releases. The group-keyed
      legacy path keeps today's `poll` semantics.
- [ ] 1.6 Have `MaxParallelRequestsLimit.release` publish a
      `slot_freed` event on Redis pub/sub channel
      `litellm:slots:{deployment_id}`. In the single-worker in-memory
      case, replace the pub/sub channel with an `asyncio.Condition` per
      deployment id.

### `litellm/router_rust/invoke/endpoints.py`

- [ ] 2.1 In `_schedule`, resolve a healthy deployment via
      `select(model, dict(request), kind="healthy")` before
      `wait_for_turn`. Thread that deployment id into the `FlowItem`.
      If `select` returns empty, build the `FlowItem` with
      `deployment_id=None` so the request waits on the group-keyed
      queue.
- [ ] 2.2 Set the response header `x-litellm-queued: True` on the
      success path for either queue keying.
- [ ] 2.3 Set the new response header
      `x-litellm-queued-key: <model_name>[:<deployment_id>]` so an
      operator can audit which keying a request used.

### `litellm/router_utils/client_initalization_utils.py`

- [ ] 3.1 No change to `MaxParallelRequestsLimit` itself, but
      `release` now publishes the `slot_freed` event (task 1.6).

## Remediation 1: proxy gate deferral

### `litellm/proxy/hooks/parallel_request_limiter_v3.py`

- [ ] 4.1 Add a new knob `defer_parallel_enforcement: str | None`
      (a model-group allowlist). When the request's `model_group` is
      in the list, `should_rate_limit` skips the parallel-count branch
      and stashes a `pending_parallel_slot` marker on the
      `ParallelSlotAcquisition`. When the model group is not in the
      list, the gate keeps its current fail-fast 503.
- [ ] 4.2 Add a new `DeferredParallelSlotCheck` that runs on the router
      side immediately after `wait_for_turn` returns True and before
      `run()`. It calls back into the proxy's `parallel_count` gauge
      and raises the same `ServiceUnavailableError` if the slot is
      taken. The 429/503 the client sees on a deferred miss is
      identical to today's, but it fires only when a slot is actually
      unavailable at the moment of admission, not at pre-call peak.
- [ ] 4.3 Document the operational stopgap: configure
      `max_parallel_requests` on the proxy gate to a value above the
      expected queue depth for that model group and accept that the
      gate becomes advisory for queued traffic.

## Regression tests

### `tests/unit/`

- [ ] 5.1 Per-deployment keying, unit (heap separation): a
      `Scheduler` with a `deployment_id`-keyed `FlowItem` places the
      entry in a separate heap from another deployment in the same
      group. A release on deployment A wakes only deployment A's
      poller. Assert heap separation and targeted wakeup.
- [ ] 5.2 Per-deployment keying, unit (slot-aware poll): `poll` for
      the per-deployment path is called while the target deployment's
      `in_flight` is at its `max_parallel_requests`, it returns False.
      The moment `in_flight` drops below `max_parallel_requests` it
      returns True, regardless of where the entry sits in the group
      heap.
- [ ] 5.3 Proxy gate deferral, unit: a fake proxy limiter returns
      `OVER_LIMIT` on pre-call when the gate is hard. The same
      limiter, with `defer_parallel_enforcement` covering the model
      group, returns `OK` on pre-call and raises only when the
      deferred check fires. Assert the deferral path.
- [ ] 5.4 Backward compatibility: the legacy
      `x-litellm-request-prioritization-used` header is still set on
      the group-keyed path. No new header is set unless the caller
      opts in. Existing priority-scheduler callers are not migrated
      automatically.

### `tests/e2e/`

- [ ] 6.1 Per-deployment keying, integration (mixed saturation): a
      group has deployment A (`max_parallel_requests=1`) and B
      (`max_parallel_requests=10`). 12 concurrent requests all target
      the group and no explicit priority is set. Under the
      group-keyed legacy path the first 12 to pop in priority order
      complete (B absorbs the load). Under the per-deployment path
      with `deployment_id` set, the requests that resolve to A queue
      behind A's single slot while the requests that resolve to B
      complete as B's 10 slots open. `x-litellm-queued-key` reflects
      the deployment id.
- [ ] 6.2 Proxy gate deferral, integration: two requests race into a
      group whose gate limit is 1 and whose queue limit is 2. Under
      variant A the second request queues and completes after the
      first releases, with `x-litellm-queued: True` set. Under
      variant B it 503s. The deferred miss path returns 429/503 with
      no `x-litellm-queued` header.
- [ ] 6.3 Two-worker proxy with shared Redis: the QA runbook in
      `AGENTS.md` applies. The `DeferredParallelSlotCheck` calls back
      into the proxy's `parallel_count` gauge from the router side,
      which is a new call boundary that needs to be exercised with a
      two-worker proxy and a shared Redis instance.

## Open questions

- Should streaming requests be added to the `unqueueable` list?
  Today streaming is not in that list (see
  `litellm/router_utils/request_priority.py`). Confirm.
- `drop_params` interaction: when `drop_params` is true, a bad
  `priority` value is coerced to the router's `default_priority`
  instead of raising, and `priority_header_fields` returns an empty
  mapping so no header is emitted. Confirm whether the new
  `x-litellm-queued` / `x-litellm-queued-key` headers should follow
  the same `drop_params` behavior or always be emitted when the queue
  is used.
