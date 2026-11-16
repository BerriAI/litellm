## Compat names

- A change here that removes a name outside code can import keeps the name only when one of these holds
  - It has no leading underscore and shipped in a stable release
  - It is private, code outside this repo calls it, and the compat keeps that code doing what it did
- A private name whose old behavior cannot be kept stays removed. An `ImportError` names what changed, while a no-op that answers wrongly hides it
- A vendored copy of litellm in another repo is not a caller

## get_model_cost_map.py

- This module decides when to fetch and install. Fetching, parsing, integrity checks and alias expansion run in Rust
- The install order is publish, `add_known_models`, router replay, then `model_cost_inputs.reapply`
  - Registrations land by lookup in the published table, so they are written after publishing
- Startup and `register_model(url)` fetch with `use_http_settings=False`, which reads env vars only like main's bare httpx. Reload passes `use_http_settings=True`, so Rust layers the configured HTTP settings on top
- Startup falls back to the bundled map on any failure. Reload never falls back and keeps the current map
- With `LITELLM_MODEL_COST_MAP_CACHE_DIR` set, startup keeps every remote document it installs and starts from the kept copy without waiting for the network. Rust owns the file, its 24 hour limit and the atomic write. This module only decides when to read, store and revalidate
  - The kept copy goes through `model_cost_inputs.install` like a fetched one. A rejected copy falls through to the blocking fetch
  - The background check sends the kept ETag. A 304 renews the copy, a new document is installed and published as a new `litellm.model_cost` once the import has finished, then stored
  - Bundled-only processes (`LITELLM_LOCAL_MODEL_COST_MAP`, offline mode, the CLI entry points) never read or write the directory. Reload does not write it either
- Reload sends the installed document's ETag when that document came from the same URL. A 304 is `ModelCostMapNotModified`, and nothing is installed or published
- `register_model(url)` makes one attempt and registers nothing on failure
- `utils._register_on_current_map` reruns the whole pass on the new table after `NativeTableSwapped`, and its last attempt passes `rows=None`, so no entry is lost
- Compat names exist only so code written against main keeps working. New code never calls them, and they never gain behavior of their own
  - `_finalize_model_cost_map`, use `model_cost_inputs.install`
  - `git_blob_id`, which Rust computes for `source_revision`
  - `FALLBACK_GENERALIZATIONS_KEY`, `RESERVED_TOP_LEVEL_KEYS` and `RETRYABLE_FETCH_STATUS_CODES`, public constants. Rust owns the retry rule
  - `GetModelCostMap.validate_model_cost_map` and `_check_*`, since Rust checks fetched documents
  - `ModelCostMapReloaded.model_cost_map` and `ModelCostMapNotModified.model_cost_map`, use `GetModelCostMap.loaded_model_cost_map()`
  - the `sleep`, `rng` and `client` parameters, which are ignored with a `DeprecationWarning`
  - `AliasedCostEntry` re-exported here, which now lives in `model_cost_table.py`

## model_cost_table.py

- This module owns the live `litellm.model_cost` table. Python code reads and writes it here, and `rust_bridge/host/model_cost.py` calls it for Rust
- `publish`, `write` and `changed_in_place` are the only writers. Each moves `ModelCostVersion`
- The version is `(id(table), len(table), writes)`, so replacing the dict or changing its size moves it without a writer
  - An in-place value edit does not, so code that edits rows in place calls `changed_in_place()`
- Anything derived from the table is cached under the version (`@VersionCached`, `model_cost_version()`), never cleared by hand
- Compat names exist only so code written against main keeps working. New code never calls them, and they never gain behavior of their own
  - `utils._invalidate_model_cost_lowercase_map` forwards to `changed_in_place`
  - `utils.get_model_cost_mutation_generation` forwards to `model_cost_version().writes`
  - `utils._model_cost_lowercase_map` is always `None`
  - `utils._runtime_registered_model_cost` is always empty, since kept registrations live in `model_cost_inputs`
