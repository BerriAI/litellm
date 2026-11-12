# Shared state across proxy workers

A production proxy runs many workers, each with its own Router built from the
same config. They agree on routing through Redis. This page explains what is
shared, what is not, and how fresh each piece is guaranteed to be.

![Three proxy workers, each with a Router and its own in-memory DualCache tier,
all syncing with one shared Redis that holds cooldowns, fail counters, usage and
strategy state](../img/shared-state-1.svg)

## The routing cache

The router keeps its routing state in a `DualCache`
(`litellm/caching/dual_cache.py`) built from the `redis_*` arguments. The cache
has two tiers, an in-memory tier and a Redis tier. Writes go to both. Reads
check memory first, go to Redis only on a miss, and then copy the Redis value
into memory. Without Redis settings, the router runs with the in-memory tier
alone. That is fine for one process, but it means every process routes on its
own view.

The cache is returned from `router.health().store`.

Reading memory first keeps Redis off the hot path, at the cost of freshness. A
value another process wrote is seen only once the local copy has expired, or
when there never was a local copy. The router handles this differently for each
kind of state.

| State | Key | Tiers | How fresh other processes see it |
| --- | --- | --- | --- |
| Cooldown | `deployment:{id}:cooldown` | memory + Redis | the local copy is re-checked against Redis at least every `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` (1s) |
| Allowed fails counter | `deployment:{id}:allowed_fails` | memory + Redis, incremented in Redis | fleet-wide at increment time, falling back to the local count if Redis fails |
| Success / fail counts for the error-rate rule | `{id}:successes`, `{id}:fails` | memory only, 60s TTL | never, each process judges on its own traffic |
| Usage per minute | `global_router:{id}:{model}:tpm:{HH-MM}`, `...:rpm:...` | Redis increment after each response | at once in Redis. A worker's copy catches up on its next write or the next minute |
| Strategy state (usage-based v2, latency, least-busy) | strategy keys | memory, pushed to Redis every `DEFAULT_REDIS_SYNC_INTERVAL` (1s) or on write | about a second behind |

See [Routing state and cache keys](../reference/state-and-cache-keys.md) for the
full list with TTLs.

## What this means in practice

Cooldowns are shared. When one worker cools a deployment down, the others stop
picking it within about a second.

Rate limits through `rpm` and `tpm` are soft. Processes do not reserve capacity
before a call. They read counters that are incremented after responses, so a
burst can overshoot a deployment's limit by roughly what the fleet sends in one
sync interval. The proxy's own rate limiter for keys and teams is a separate
system that uses atomic Lua scripts, and the router does not use it.

The default error-rate cooldown is decided per process. A deployment that fails
half its traffic everywhere gets cooled down by each process independently, and
the resulting cooldown key is shared. With `allowed_fails` or an
`allowed_fails_policy` set, the counter is in Redis and the threshold is
fleet-wide.

![Sequence in which one worker sets a deployment cooldown in Redis after a 429,
a second worker reads it and filters the deployment too, and the key expires on
both tiers](../img/shared-state-2.svg)

## Redis outages

The router keeps routing when Redis fails. The cache logs the failure and serves
from memory, and the `allowed_fails` increment falls back to the process's own
count. Routing quality drops to what each process knows alone until Redis
returns.
