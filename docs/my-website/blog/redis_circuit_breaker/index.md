---
slug: redis-circuit-breaker
title: "Making the AI Gateway Resilient to Redis Failures"
date: 2026-04-11T09:00:00
authors:
  - ishaan
description: "How LiteLLM's production AI Gateway handles Redis degradation at scale without cascading failures — circuit breaker pattern, 0ms fast-fail, automatic recovery."
tags: [reliability, redis, infrastructure, engineering, ai-gateway]
hide_table_of_contents: true
---

import { CascadeFailure, CircuitBreakerStates, CircuitBreakerFlow, IncidentTimeline } from './diagrams';

*Last Updated: April 2026*

Enterprise AI Gateway deployments put Redis in the hot path for nearly every request: rate limiting, cache lookups, spend tracking. When Redis is healthy, the latency contribution is single-digit milliseconds — invisible to end users. When it degrades, a production AI Gateway needs to stay up regardless.

Running LiteLLM at scale across 100+ pods means designing for failure modes before they appear. The easy case is Redis going fully down: fail fast, fall through to the database, continue serving requests. The hard case — the one that takes down gateways — is a *slow* Redis: still accepting connections, still responding, but timing out after 20-30 seconds per operation.

{/* truncate */}

## Why slow Redis is harder than a full outage

<CascadeFailure />

With 100 pods each hanging 30 seconds on every auth check, threadpools fill up and requests queue. By the time Redis times out and falls through to Postgres, the database receives 100× its normal load from simultaneous fallbacks. A slow Redis becomes a database outage becomes a full gateway outage. A production-grade AI Gateway cannot allow one degraded dependency to cascade into total failure.

## The fix: circuit breaker pattern

The circuit breaker pattern tracks consecutive failures and cuts off the unhealthy dependency before it cascades. Instead of hanging 30 seconds on each Redis call, the circuit opens after 5 consecutive failures and fast-fails at 0ms — no network call, no wait.

<CircuitBreakerStates />

Three states:

- **CLOSED** — normal. All Redis calls pass through.
- **OPEN** — Redis is unhealthy. Every call fast-fails instantly. Requests continue with degraded-but-functional behavior: auth and rate limiting fall back to the database.
- **HALF-OPEN** — after 60 seconds, one probe request tests recovery. Success closes the circuit; failure resets the timer.

This is how a reliable AI Gateway handles infrastructure degradation: stay up, degrade gracefully, recover automatically.

## How requests flow through the AI Gateway

<CircuitBreakerFlow />

When the circuit is open, the gateway does not stall. Auth checks fall back to Postgres — slower, but bounded. The database absorbs the load because it receives *some* requests via DB fallback, not *all* 100 pods simultaneously dumping their queued requests after a 30-second timeout.

The difference between a resilient AI Gateway and a fragile one: controlled degradation vs. uncontrolled cascade.

## The implementation

```python
class RedisCircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: int):
        self.failure_threshold = failure_threshold  # default: 5
        self.recovery_timeout = recovery_timeout    # default: 60s
        self._failure_count = 0
        self._state = self.CLOSED

    def is_open(self) -> bool:
        if self._state == self.OPEN:
            if time.time() - self._opened_at > self.recovery_timeout:
                self._state = self.HALF_OPEN
                return False  # this caller is the recovery probe
            return True       # fast-fail
        return False

    def record_failure(self):
        self._failure_count += 1
        self._opened_at = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN  # open the circuit

    def record_success(self):
        self._failure_count = 0
        self._state = self.CLOSED   # Redis recovered
```

Every async Redis operation goes through a decorator that checks the breaker before touching the network. When open, it raises immediately:

```python
@_redis_circuit_breaker_guard
async def async_get_cache(self, key: str):
    ...
```

The decorator handles all bookkeeping — success resets nothing, failures increment the counter, exceptions trigger `record_failure()`. The caller sees a clean exception and falls through to its normal non-Redis path. No changes required in calling code.

## AI Gateway resilience in production

<IncidentTimeline />

Redis degradation events no longer cascade in production. The observable symptom during a Redis slowdown is a temporary bump in cache miss rate — the right failure mode for a resilient AI Gateway. Auth still works. Rate limiting still works. Spend tracking still works, at slightly higher DB cost. Recovery is fully automatic when Redis comes back.

```bash
# configure via environment variables
REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5   # failures before opening
REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60  # seconds before probe
```

The circuit breaker ships on by default in all LiteLLM versions since `v1.82.0`. No configuration needed for most deployments.

## Key Takeaways

- A slow Redis is more dangerous than a downed one: 30-second timeouts across 100+ pods overwhelm Postgres at 100× normal load
- LiteLLM's AI Gateway uses a circuit breaker that fast-fails Redis calls at 0ms after 5 consecutive failures
- Three states: CLOSED (normal), OPEN (fast-fail + DB fallback), HALF-OPEN (probe recovery)
- Auth, rate limiting, and spend tracking continue working during Redis outages
- Resilient, production-grade behavior — enabled by default since `v1.82.0`, no configuration required

---

### Frequently Asked Questions

### Does the circuit breaker affect normal Redis performance?

No. When Redis is healthy (circuit CLOSED), every call passes through with zero overhead. The breaker only activates after 5 consecutive failures — transparent under normal conditions.

### What happens to rate limiting when the circuit is open?

Rate limiting falls back to Postgres with bounded load. Limits remain enforced at slightly higher DB cost until Redis recovers and the circuit closes automatically.

### How is this different from basic Redis retry logic?

Retry logic still waits for each timeout (30s × retries). The circuit breaker cuts the connection immediately at 0ms after the failure threshold, preventing threadpool exhaustion across all pods simultaneously. Retries make slow-Redis worse; the circuit breaker contains it.

### Is this available in LiteLLM OSS?

Yes. The circuit breaker ships in LiteLLM OSS (Apache 2.0) by default since `v1.82.0`. [LiteLLM Enterprise](https://litellm.ai/enterprise) adds SSO/SCIM, air-gapped deployment, 24/7 SLA support, and advanced guardrails on top of the OSS foundation.

---

## Conclusion

Redis resilience is one layer of what makes LiteLLM a production-grade, reliable AI Gateway at scale. The circuit breaker pattern ensures infrastructure degradation stays contained — the right failure mode is a temporary cache miss rate bump, not a full outage. This is how AI Gateway infrastructure should behave under pressure: degrade gracefully, recover automatically, keep serving traffic. For teams with strict uptime and compliance requirements, [LiteLLM Enterprise](https://litellm.ai/enterprise) provides the additional controls needed for regulated production environments.

## Recommended Reading

- [LiteLLM AI Gateway — full feature overview](https://docs.litellm.ai/docs/simple_proxy)
- [Load balancing and routing across 100+ LLM providers](https://docs.litellm.ai/docs/routing)
- [Spend tracking and budget controls](https://docs.litellm.ai/docs/proxy/cost_tracking)
