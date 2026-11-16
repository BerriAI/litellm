## Scope

- This crate is the typed data model of a config and reads no file. Reading files and `include` live in `config-load`, which hands the merged document to `Config::from_value`

## Redis settings

- This crate only deserializes YAML. `RedisSettings` is the raw `coordination_redis` or `cache_params` map and `CacheSettings` tells a `type: redis` cache apart from the rest. Reading those maps the way `litellm/_redis.py` does belongs to `gateway/src/redis`, and turning merged kwargs into a `Resolution` belongs to `redis-resolve`

## Route settings

- `RouteList` is the typed form of a route list in config and key metadata, and both hosts read through it. A value that is not a list of strings is `RouteList::Malformed`, which grants nothing. A route template inside such a list is starlette syntax, so it is parsed by `litellm-starlette`, never here
