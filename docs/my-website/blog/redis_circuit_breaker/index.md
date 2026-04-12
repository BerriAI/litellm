---
slug: redis-circuit-breaker
title: "Making the AI Gateway Resilient to Redis Failures"
date: 2026-04-11T09:00:00
authors:
  - ishaan
description: "We built a circuit breaker into LiteLLM so a slow Redis never cascades into a gateway outage. Here's how it works."
tags: [reliability, redis, infrastructure, engineering]
hide_table_of_contents: true
---

import { CascadeFailure, CircuitBreakerStates, CircuitBreakerFlow, IncidentTimeline } from './diagrams';

Redis is in the hot path for almost every request through LiteLLM: rate limiting, cache lookups, spend tracking. When Redis is healthy, the latency contribution is single-digit milliseconds. When it degrades, you need a plan - not just for when Redis is fully down, but for when it's *slow*.

Running an AI gateway at scale across 100+ pods means designing for failure modes before they show up in production. The dangerous case is not Redis being fully down. A complete outage is easy to handle - fail fast, fall through to the database, continue. The dangerous case is a *slow* Redis: still up, still accepting connections, but timing out after 20-30 seconds on each operation.

{/* truncate */}

## Why slow Redis is harder than down Redis

<CascadeFailure />

With 100 pods each hanging for 30 seconds on every auth check, the threadpool fills up. Requests queue. By the time Redis times out and falls through to Postgres, the database is receiving 100x its normal load from simultaneous fallbacks. A slow Redis becomes a database outage becomes a full gateway outage.

## The fix: circuit breaker

The circuit breaker pattern solves this by tracking consecutive failures and cutting off the unhealthy dependency before it can cascade. Instead of hanging for 30 seconds on each Redis call, the circuit opens after 5 consecutive failures and fast-fails immediately - 0ms, no network call.

<CircuitBreakerStates />

Three states:

- **CLOSED** - normal. All Redis calls pass through.
- **OPEN** - Redis is unhealthy. Fast-fail every call instantly. The request continues with degraded-but-functional behavior (DB fallback for auth).
- **HALF-OPEN** - after 60 seconds, one probe request is allowed through to test recovery. Success closes the circuit; failure resets the timer.

## How requests flow through it

<CircuitBreakerFlow />

When the circuit is open, the gateway does not stall. Auth checks fall back to Postgres - slower, but bounded. The database can handle the load because it is receiving *some* requests via DB, not *all* requests via DB simultaneously after 100 pods each waited 30 seconds for Redis to fail.

The difference: controlled degradation vs. uncontrolled cascade.

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

The decorator handles the bookkeeping - success increments nothing, failure increments the counter, exceptions trigger `record_failure()`. The caller sees a clean exception and falls through to its normal non-Redis path.

## What this looks like in production

<IncidentTimeline />

Redis degradation events no longer cascade. The observable symptom during a Redis slowdown is a temporary bump in cache miss rate - the right failure mode. Auth still works, rate limiting still works (at slightly higher DB cost), and recovery is fully automatic when Redis comes back.

```bash
# configure via environment variables
REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5   # failures before opening
REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60  # seconds before probe
```

The circuit breaker is on by default in all LiteLLM versions since `v1.82.0`. No configuration needed for most deployments.
