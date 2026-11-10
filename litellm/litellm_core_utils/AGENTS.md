## get_model_cost_map.py

- This module decides when to fetch and install. Fetching, parsing, integrity checks and alias expansion run in Rust
- The install order is publish, `add_known_models`, router replay, then `model_cost_inputs.reapply`
  - Registrations land by lookup in the published table, so they are written after publishing
- Startup and `register_model(url)` fetch with `http_settings=None`, which reads env vars only like main's bare httpx. Reload passes `http_settings()`
- Startup falls back to the bundled map on any failure. Reload never falls back and keeps the current map
- `register_model(url)` makes one attempt and registers nothing on failure
- `utils._register_on_current_map` reruns the whole pass on the new table after `NativeTableSwapped`, and its last attempt passes `rows=None`, so no entry is lost
- Compat names exist only so code written against main keeps working. New code never calls them, and they never gain behavior of their own
  - `_finalize_model_cost_map`, use `model_cost_inputs.install`
  - `GetModelCostMap.validate_model_cost_map` and `_check_*`, since Rust checks fetched documents
  - `ModelCostMapReloaded.model_cost_map`, use `GetModelCostMap.loaded_model_cost_map()`
  - the `sleep`, `rng` and `client` parameters, which are ignored with a `DeprecationWarning`
  - `AliasedCostEntry` re-exported here, which now lives in `rust_bridge/host/model_cost.py`
