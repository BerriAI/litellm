## Grouping lives in the types

The YAML shape is Python's `config.yaml` and we never change it, so a mapping stays flat even when its keys serve different consumers. The grouping is encoded in the types instead, with flattened structs (see the "Serde flatten" rules in `../AGENTS.md`)

### Rules

- Every distinct set of readers gets its own flattened struct, even when only one key has that set. `namespace` in `cache_params` is read only by `redis`, so it gets its own struct and does not sit with `host` and `port`, which `redis-semantic` and `valkey-semantic` also read
- Only keys that every reader of the mapping uses stay on the parent struct. A key that some readers ignore never stays on the parent just because no group fits yet. Make the group
- Once a mapping has keys with different readers, all of them move into groups, never only some. `router_settings` typed as one flat struct, or a parent that keeps keys an existing group already covers, breaks this

### Finding the readers

- A reader is the code that receives or branches on the value, found in the Python that consumes the key. The key's name does not tell you. `default_in_redis_ttl` looks Redis-only but Valkey reads it too, and `default_redis_ttl` sits under `cache_params` yet configures the proxy's internal caches, not the response cache
- Follow the value to where it is used, not just to where it is passed. A constructor argument read in one `if type == ...` arm has that arm as its reader. A value applied behind an `isinstance` guard is read by that class and its subclasses only (`namespace` reaches `RedisCache` and `RedisClusterCache`, not `RedisSemanticCache`)
- A key read by the shared code path for every variant (e.g. `ttl` and `mode`, read by `Cache` itself) is read by all of them and stays on the parent

### Naming

- Name each struct for its reader set (`RedisServerCacheParams`, `SemanticCacheParams`), and name the parent's flattened field for the same set (`redis_server`, `semantic`)
- When the name alone does not say who reads the keys, add a one-line doc comment on the struct that lists the readers, like the one on `RedisServerCacheParams`

### Tests

- Each mapping has one rstest with a case per group, putting one key of that group in the YAML and asserting it lands in that group's field and not in `additional_fields`
- When a key's readers were easy to get wrong (the examples above), add a case for that key too, so moving it back into a broader group fails a test

## Lists are boxed slices

Config is parsed once and only read after that, so every list field on a config type is `Box<[T]>` (or `Option<Box<[T]>>`), never `Vec<T>`. The type then says the list can't grow, and serde deserializes into it directly

- `Vec<T>` is only for values that get built up and thrown away, like the layers and origins in `include/`. Those never become config fields
- Code that builds a field by hand collects into a `Vec` and converts at the end (`Box::from`, `.into_boxed_slice()`), and never keeps the `Vec` around
