## Redis settings

- This crate only deserializes YAML. `RedisSettings` is the raw `coordination_redis` or `cache_params` map and `CacheSettings` tells a `type: redis` cache apart from the rest. Reading those maps the way `litellm/_redis.py` does belongs to `gateway/src/redis`, and turning merged kwargs into a `Resolution` belongs to `redis-resolve`
