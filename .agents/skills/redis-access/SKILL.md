---
name: redis-access
description: Add Redis operations or move raw redis-py clients and scripts behind RedisTransport in litellm/redis, without porting them to Rust or expanding migration scope
metadata:
  internal: true
---

# Redis access

Python reaches Redis through `RedisTransport` (`litellm/redis/transport/contract.py`). Read `references/transport.md` for adding a command, the failure kinds, the namespace and TTL rules, and moving raw `init_async_client` / `async_register_script` / `.redis_client` access onto the transport

Rules that always apply

- New code never calls `init_async_client`, `init_pubsub_client`, `async_register_script` or `.redis_client`, and never imports redis-py outside `litellm/redis/transport/redis_py.py`
- Keys and values in Redis are a wire format shared with older pods and Rust. A move keeps them byte for byte
- A `transport` failure on a write is an unknown outcome, never "not applied"
- Do only what was asked. Confirm before any Rust work
