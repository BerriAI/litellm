- `fsm` is a generic state machine runtime and knows nothing about breakers. `breaker` is a generic circuit breaker and knows nothing about Redis. Deciding which errors count as failures belongs to the consumer
- `Machine::step` stays pure. `&self` is config and time comes in on the event, never read inside the machine. The only clock read is `Admission`'s drop, at the call-site boundary
- Transition rules are one `match (state, event)` per machine, with no macro DSL and no `_` arm that would hide a new state from the compiler
- `Shared` counts a transition, bumps the generation and calls the observer only when the state's enum variant changes, matching Python bumping `generation` only when `_state` changes. Data inside a variant (failure counters, `since`) changes without a transition
- The observer runs under the lock so transitions are seen in generation order. Keep it to logging or a metric
- A disabled breaker is `Option<Breaker>::None` at the consumer, not a flag inside the machine. `Config` comes in already resolved, reading `REDIS_CIRCUIT_BREAKER_*` env is the consumer's job

## Parity with Python `RedisCircuitBreaker` (`litellm/caching/redis_cache.py`)
- The rstest table `matches_the_python_redis_circuit_breaker` was checked case by case against the Python breaker. A behavior change here needs the same check, by driving the Python class with the same steps
- Divergences, each on purpose
  - monotonic `Instant` instead of `time.time()`, so a wall clock jump can't open or close the breaker
  - a probe `Admission` dropped without an outcome records a hard failure and reopens. Python's probe that dies on `CancelledError` (a `BaseException`) records nothing and leaves the breaker HALF_OPEN for good, and a dropped Rust future is common
  - `_failure_count` is dropped, it only fed a log line
  - no `enabled` flag, see above
