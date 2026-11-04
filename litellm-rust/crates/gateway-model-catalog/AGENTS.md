# Gateway model catalog

The native gateway's own cost map, loaded and refreshed with the same Rust code the Python proxy uses through python-bridge. Fetching, retry, `Retry-After`, status classification and provenance come from `model-catalog-refresh`, parsing and the size check from `model-catalog`. Nothing here reimplements them

- One `ArcSwap<Installed>` holds the catalog and its provenance together, so a reader always sees one install. Swap a whole `Installed`, never one half
- Boot parses `BUNDLED_REGISTRY` and never waits on the network. `spawn_revalidate` then fetches the configured source once in the background and replaces the boot map only if nothing else was installed meanwhile
- A failed fetch, a document that does not parse and a map that fails the size check all keep the installed catalog. The background refresh records why in `fallback_reason`, the way Python's `_keep_bundled_after` does
- Rows that fail validation are left out of a fetched map before the size check runs, so the gateway serves typed rows only. Python serves those rows as written
- Settings are the Python process variables, `LITELLM_LOCAL_MODEL_COST_MAP`, `LITELLM_OFFLINE_MODE`, `LITELLM_MODEL_COST_MAP_URL`, `MODEL_COST_MAP_MIN_MODEL_COUNT` and `MODEL_COST_MAP_MAX_SHRINK_RATIO`, parsed as `settings.rs` pins. Config `environment_variables` do not apply, since Python reads these at import
- The HTTP client comes from the gateway's `HttpClientPool`, `NoRedirect` like python-bridge's. Shutdown drops an in-flight fetch or retry wait through the injected `CancellationToken`
- Deployment pricing is never written into the catalog. There is no `register_model` here, a deployment override belongs at price time

## Routes

`router` serves `/reload/model_cost_map` and `/model/cost_map/source` with the response fields of the Python routes in `proxy_server.py`, and the gateway mounts it behind its admin authentication. Errors use FastAPI's `{"detail": message}`. `/schedule/model_cost_map_reload` stays out until the gateway reads the proxy DB, which is also why a reload publishes no revision for other pods

## Tests

Lifecycle tests live in `catalog.rs` and drive real `file://` and wiremock sources through `test_support`, whose size limits accept small documents. Route tests in `routes.rs` call the router with `oneshot`. Auth on these routes is tested where they are mounted, in `crates/gateway/tests/server.rs`
