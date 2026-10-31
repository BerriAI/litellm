## Concepts

Every price in a row lowers to one `Price { meter, channel, when }` and its `UnitPrice`. These keys come from the `FlatKey::parse` tests

- `input_cost_per_token`
  - meter `Input(Token(Text))`, channel `Online`, tier `Standard`, band `Base`
- `output_cost_per_token_above_272k_tokens_priority`
  - meter `Output(Token(Text))`, channel `Online`
  - `_priority` sets tier `Priority`
  - `_above_272k_tokens` sets band `OverPromptTokens(272000)`
- `cache_creation_input_token_cost_above_1hr_above_200k_tokens`
  - meter `CacheWrite { Text, OneHour }`, where `_above_1hr` is the cache TTL, not a band
  - `_above_200k_tokens` sets band `OverPromptTokens(200000)`
  - channel `Online`, tier `Standard`
- `ocr_cost_per_page_batches`
  - meter `Feature(OcrPage)`, tier `Standard`, band `Base`
  - `_batches` sets channel `Batch`

- Meter is what gets counted, such as text input tokens, reasoning tokens, an image of a given size or a web search
  - A variation only one meter has stays inside it (cache TTL, image size, resolution, `_above_Ns_interval`, `max_results_range`)
- Band is a range of prompt sizes
  - `_above_200k_tokens` prices a call whose prompt is over 200k tokens, and `Band::Base` is everything under the lowest edge
  - `Prices::band` puts the whole call in one band from its prompt token count
  - `Billing::band_edge` says whether a prompt of exactly 200k is in that band
- `When` is the request's condition, its service tier and band
- `Channel` is the billing route
  - Batch and decisions are channels and not tiers because Python bills them with their own rules
- Off-peak and `Adjustments` are never prices
  - Off-peak swaps in other token rates while a UTC window is open
  - `Adjustments` multiplies a priced call for region, residency, geo or `speed: fast`
- `Billing` on `Prices` holds row attributes that change how its prices apply (text unit, band edge, cache-read fallback, wall clock, web search unit, tier scope)
  - `lower.rs` derives it from the row and its `litellm_provider`, so `litellm-cost` holds no provider list

## Layout

`wire/` reads one entry as written. `EntryWire::read` makes one pass over the entry's keys and puts every non-null key in exactly one place, which is an attribute, a flat price parsed by `FlatKey::parse`, a price block, a `supports_*` capability or `unknown`. A known key with a bad value becomes an `Error` that names the key, and the rest of the entry is still read

`lower.rs` is the only code that knows both the wire types and `litellm-cost`. It lowers an `EntryWire` into `Prices`, `Adjustments` and `Model` and checks the rules that span keys, such as one price per meter and condition, contiguous tiers, no suffix inside a block, and off-peak needing hours and prices. Lowering only goes one way. Nothing turns `Prices` back into JSON, since the raw fields are kept

`catalog.rs` reads the document, indexes aliases and runs the checks across entries. A bad entry keeps its raw fields and its errors are listed in `Loaded.errors`. Deciding what to do with them is the caller's job. `Catalog::written` gives back every top-level key and value as the document wrote them, in document order. `Catalog::served` gives back every model and alias as Python serves it, without `sample_spec` and `fallback_generalizations`

`store/` is the catalog's in-memory store and the only place that indexes model names. Its rules are in [src/store/AGENTS.md](src/store/AGENTS.md)

`layers.rs`, `resolve/`, `model_info.rs` and `register.rs` hold types only. The functions over them land with their first caller

Catalog errors use one opaque `Error` with typed causes and JSON Pointer accessors (`path`, `key`, `field`). `Error::rejects` alone decides whether an error fails the document, rejects its row or neither, and `ErrorKind` is the stable name hosts report it by

## Prices

The price vocabulary lives in `litellm-cost`, and only this crate knows JSON key names

A new `N`, image size or resolution in an existing key shape parses with no Rust change. A new unit, token kind, feature or tier needs a variant in `litellm-cost` plus its spelling in `wire/flat_key.rs` and `FLAT_KEY_PATTERN` in `schema.rs`

`Prices::band` and `Prices::rate` are pinned to Python's `_get_token_base_cost` by `tests/price_lookup_reference.json`. After a billing change in Python, regenerate it from the repo root with `PYTHONPATH=. python litellm-rust/crates/model-catalog/tests/generate_price_lookup_reference.py`

## Tests

`BUNDLED_REGISTRY` embeds the root `model_prices_and_context_window.json`, so the tests over it fail a JSON change with a bad value or a key no type reads

`Catalog::problems` is the one definition of a mergeable cost map. The bundled test asserts it is empty, and the cost map guard workflow runs it on a pull request's JSON through `cargo run --bin check-registry <path>`
