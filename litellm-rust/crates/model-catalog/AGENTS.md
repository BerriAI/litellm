## Catalog keys

`Model::from_fields` reads a catalog entry and sorts every non-null key into exactly one part. `attributes` holds the named non-price fields, `prices` holds every price, `capabilities` holds every boolean `supports_*` key, and `unrecognized` holds the keys no type of this build reads

A known key with a bad value is a `Problem` that names the model, the key, what was expected and what was received. `Catalog::parse` and `ServedCatalog::register` collect every problem and reject the whole document or registration, so a bad value fails where it is written (a registry fetch, `register_model`, a deployment load, a request's custom pricing) and never while a call is being billed. An unknown key is not a problem. It stays in `ModelEntry::fields`, which is what `litellm.model_cost` exposes to Python

## Validation

For `model_prices_and_context_window.json` validation, we should eventually:

- Stop skipping this crate's tests

## Prices

Every price is `PriceKey -> Usd`, where `PriceKey` is the `Meter` it counts plus the circumstances it applies under (`tier`, `band`, `clock`). A variation only one meter has lives in the meter (cache TTL, image size, resolution, search context size). A variation that spans meters is a `PriceKey` field. Regional uplifts are `Multiplier`s kept beside the prices, never prices themselves

`Prices::lower` reads all five JSON spellings into that one key space, so two spellings of the same price are a `DuplicatePrice` problem. Flat keys follow the grammar in `src/rate.rs`, `{input|output|feature}_cost_per_{unit}`, `cost_per_{unit}`, `cache_read_input_{token kind}_cost` or `cache_creation_input_{token kind}_cost[_above_1hr]`, then an optional `_above_{N}k_tokens` or `_above_{N}s_interval`, then an optional `_{tier}`. Keep that suffix order. A new price key that fits the grammar needs no Rust change. A key in a new shape needs a grammar change in `rate.rs` (and `key_pattern`) or a named field in `Attributes` or `PricingFields`
