# Cache boundary

This folder owns how Rust inference reaches the selected Python cache. Cache algorithms, storage protocols and response-cache semantics belong to their cache crates

`mod.rs` exposes the cache boundary to routes. `selection.rs` asks `litellm.rust_bridge.host.cache` for the live configured cache and owns route admission. A route's protocol carries cache operations through any host call type that converts from `CacheCall`

`python/` delegates operations to the selected Python cache without discovering configuration. Routes use only the parent module's exports
