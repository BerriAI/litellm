## Layout

`wire/` reads one entry as written. `EntryWire::read` makes one pass over the entry's keys and puts every non-null key in exactly one place, which is an attribute, a flat price parsed by `FlatKey::parse`, a price block, a `supports_*` capability or `unknown`. A known key with a bad value becomes a `Located` error that names the key, and the rest of the entry is still read

`lower.rs` is the only code that knows both the wire types and `litellm-cost`. It lowers an `EntryWire` into `Prices`, `Adjustments` and `Model` and checks the rules that span keys, such as one price per meter and condition, contiguous tiers, no suffix inside a block, and off-peak needing hours and prices. Lowering only goes one way. Nothing turns `Prices` back into JSON, since the raw fields are kept

`catalog.rs` reads the document, indexes aliases and runs the checks across entries. A bad entry keeps its raw fields and its errors are listed in `Loaded.errors`. Deciding what to do with them is the caller's job. `Catalog::written` gives back every top-level key and value as the document wrote them, in document order. `Catalog::served` gives back every model and alias as Python serves it, without `sample_spec` and `fallback_generalizations`

`Billing` on `Prices` holds the row attributes pricing reads (text unit, band edge, cache-read fallback, wall clock, web search unit, tier scope). `lower.rs` derives it from the row and its `litellm_provider`, so `litellm-cost` holds no provider list

`layers.rs`, `resolve.rs`, `model_info.rs` and `register.rs` hold types only. The functions over them land with their first caller

Catalog errors use one `Error` enum with a `Located` JSON Pointer. Only `lower.rs` decides which errors are fatal and which reject a row

## Prices

The price vocabulary lives in `litellm-cost`, and only this crate knows JSON key names. Every price is keyed by `Meter`, `Channel` and `When`. A variation only one meter has stays in the meter (cache TTL, image size, resolution, `_above_Ns_interval`, `max_results_range`). Service tier and prompt band make up `When`. Batch and decisions are a `Channel` because Python bills them with their own rules. Off-peak is an overlay on `Prices`, and regional uplifts are `Adjustments`, never prices

A new `N`, image size or resolution in an existing key shape parses with no Rust change. A new unit, token kind, feature or tier needs a variant in `litellm-cost` plus its spelling in `wire/flat_key.rs` and `FLAT_KEY_PATTERN` in `schema.rs`

`Prices::band` and `Prices::rate` are pinned to Python's `_get_token_base_cost` by `tests/price_lookup_reference.json`. After a billing change in Python, regenerate it from the repo root with `PYTHONPATH=. python litellm-rust/crates/model-catalog/tests/generate_price_lookup_reference.py`

## Tests

`BUNDLED_REGISTRY` embeds the root `model_prices_and_context_window.json`, so the tests over it fail a JSON change with a bad value or a key no type reads
