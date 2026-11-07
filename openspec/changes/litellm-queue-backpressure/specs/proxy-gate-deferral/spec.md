# Spec delta: proxy-gate-deferral

## ADDED

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

### Requirement: operational stopgap is documented while the code change is in flight

Variant B (soft gate with a higher limit) is documented while variant A
is in flight. The operator configures the proxy gate's
`max_parallel_requests` above the expected queue depth for that model
group and accepts that the gate becomes advisory for queued traffic.

#### Scenario: operator sizes the gate above expected queue depth

- **WHEN** the proxy gate's `max_parallel_requests` is set above the
  expected queue depth for a model group
- **THEN** queued traffic does not hit the gate's 503 at pre-call peak
- **AND** the gate is advisory for queued traffic on that model group

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
