# Routing state and cache keys

Everything the router remembers between requests, where it lives, and how long
it lasts. This state lives in the router's routing cache, a `DualCache`. Every
worker uses the same keys, encodings and TTLs, so a fleet shares one Redis. In
the keys below, `{id}` is the deployment's `model_info.id` and `{minute}` is the
UTC time as `HH-MM`.

![The routing cache keys shared through Redis when it is configured, beside the
state kept only in this process](../img/state-and-cache-keys-1.svg)

| Key | Written by | Read by | Value | TTL | Tiers |
| --- | --- | --- | --- | --- | --- |
| `deployment:{id}:cooldown` | `Health::add_cooldown` | the cooldown filter, `health().cooldowns` | `{exception_received, status_code, timestamp, cooldown_time}` | the cooldown time | memory + Redis, memory re-checked against Redis every 1s |
| `deployment:{id}:allowed_fails` | `Health::record_failure` | the same | fail count | the cooldown time | memory + Redis |
| `deployment:{id}:allowed_fails:{suffix}` | the same, with a deployment-level policy | the same | fail count per error type | the cooldown time | memory + Redis |
| `{id}:successes`, `{id}:fails` | `Health::record_success`, `Health::record_failure` | the default error-rate cooldown rule, `health().failures_this_minute()` | count | 60s | memory only |
| `global_router:{id}:{model}:tpm:{minute}`, `:rpm:` | `Health::record_success` after each response | pre-call rate limit checks, `health().usage()` | count | 60s | Redis increment, memory |
| `global_router:{id}:{model}:itpm:{minute}`, `:otpm:` | the input and output token rate limit check | the same check | input and output token counts | 60s | memory + Redis |
| `{id}:{model}:tpm:{minute}`, `{id}:{model}:rpm:{minute}` | the `usage-based-routing-v2` strategy | the same | count | 60s | memory, pushed to Redis every 1s |
| `{model_group}_map` | the `latency-based-routing` strategy | the same | latency and ttft samples per deployment | `routing_strategy_args.ttl`, 1 hour | memory + Redis |
| `{model_group}_request_count:{id}` | the `least-busy` strategy | the same | requests in flight | 1 hour | memory + Redis |

Strategies and plugins keep their own keys (the strategy counters above, the
budget limiter, prompt cache pins, deployment and session affinity).

## Process-local state

Some state is per router and is never shared.

- the registry, with its indexes and `registry().version`, and the settings
  view, each held as one snapshot that `apply()` swaps
- provider clients, cached per deployment for `client_ttl` seconds
- the background health check results in `health().health_checks`

This state is rebuilt from config in every process, which is why deployment ids
must be stable across processes.
