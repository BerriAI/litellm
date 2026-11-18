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
- `UnitPrice` is USD for one unit of the meter, such as one token, image, page or second
- Band is a range of prompt sizes, starting where `_above_<N>_tokens` or `_above_<N>k_tokens` says
  - `Quote::rate` reads a meter at the highest of its own bands the prompt passes, and `Band::Base` is below every edge
  - Prompt tokens alone pick the band, so output and cache tokens bill all their tokens at the band the prompt reached
  - A meter with no price in the top band keeps its own lower band, never the base
  - `Billing::band_edge` says whether a prompt exactly at the edge is in the band (exclusive by default, inclusive for xAI)
  - A band only another tier or the batch channel prices never moves the call
- `When` is the request's condition, its service tier and band
  - Within a band the call's tier is read before standard, and any band the prompt passes is read before the base
- `Channel` is the route whose price list the call reads, and each falls back differently
  - `Online` falls back across `When` as above
  - `Batch` (`_batches`) uses bands but no service tier, and a missing price is half the online rate
  - `Decisions` (`_decisions`, OpenAI `/v1/decisions`) overrides single online fields, and a missing price is the full online rate
- Off-peak and `Adjustments` are never prices
  - Off-peak swaps in other token rates while a UTC window is open
  - `Adjustments` multiplies a priced call for region, residency, geo or `speed: fast`
- `Billing` on `Prices` holds row attributes that change how its prices apply (text unit, band edge, cache-read fallback, wall clock, web search unit, tier scope)
  - `lower.rs` derives it from the row and its `litellm_provider`, so `litellm-cost` holds no provider list

## Layout

`wire/` reads one entry as written. `EntryWire::read` makes one pass over the entry's keys and puts every non-null key in exactly one place, which is an attribute, a flat price parsed by `FlatKey::parse`, a price block, a `supports_*` capability or `unknown`. A known key with a bad value becomes an `Error` that names the key, and the rest of the entry is still read

`lower.rs` is the only code that knows both the wire types and `litellm-cost`. It lowers an `EntryWire` into `Prices`, `Adjustments` and `Model` and checks the rules that span keys, such as one price per meter and condition, contiguous tiers, no suffix inside a block, and off-peak needing hours and prices. Lowering only goes one way. Nothing turns `Prices` back into JSON, since the raw fields are kept

`catalog.rs` reads the document, indexes aliases and runs the checks across entries. A document that starts with a UTF-8 or UTF-16 byte order mark is read as that text. A bad entry keeps its raw fields and its errors are listed in `Loaded.errors`. An `alias_of` row that carries other fields is still served as its target until `without_rejected` drops it. Deciding what to do with them is the caller's job. `Catalog::written` gives back every top-level key and value as the document wrote them, in document order. `Catalog::served` gives back every model and alias as Python serves it, without `sample_spec` and `fallback_generalizations`

`store/` is the catalog's in-memory store and the only place that indexes model names. Its rules are in [src/store/AGENTS.md](src/store/AGENTS.md)

`cost_map.rs` holds a host's inputs, the installed document and the `register_model` overlay in `register.rs`. It serves the document in the key order and alias sharing Python's `litellm.model_cost` has, without the overlay, since where a registration lands is a lookup in the published table. It never holds the live table, which belongs to the host. `CostMap::read` refuses a document with a top-level value that is not a row, since a host table has nowhere to serve one

- `CostMap` is immutable behind an `Arc`. A new document is a new `CostMap`
- `Registrations<V>` is generic over the host's value type, because Python registrations can hold `datetime` and other values JSON cannot. The host keeps its own objects (`Py<PyDict>` in python-bridge) and does the merging
  - Entry order is latest-registration order. `landed_on` records which registered keys landed on each row key
- `write_pass` stops at the first `place` that returns `None` and reports `Pass::Swapped`. Retrying is the host's job

Catalog errors use one opaque `Error` with typed causes and JSON Pointer accessors (`path`, `key`, `field`). `Error::rejects` alone decides whether an error fails the document, rejects its row or neither, and `ErrorKind` is the stable name hosts report it by

`model_info.rs` holds types only. The functions over it land with its first caller

`resolve/` turns a call into the rows main tries, in main's order. Each file is a port of named Python functions and says which

- `names.rs` builds the seven names `get_model_info` tries for a known provider (`Candidates::potential`)
- `bedrock.rs` reads Bedrock model id spellings. `litellm-llms` re-exports it, so the routes and the lookups share one copy
- `cost.rs` is `Snapshot::candidates`, the `/v1/messages` Claude providers only (`CostProvider`). Any call main would price by another path is `AfterRows::Deferred` with its `Deferral`, so Python keeps pricing it. Rows before a deferral are still tried first, because main tries them first
- `cost_candidates.json` is main's answer for each test case, captured once from `completion_cost` with `NativePrices` spied. Recapture it when main's selection changes on purpose, never to make a Rust change pass

`snapshot.rs` is one call's view of a host's live table, keyed on `Generation` (main's `ModelCostVersion` plus the version of the host's installed `fallback_generalizations` rules, whose contents the host hands in with the table) and on `CostConfig`, the discount and margin config the host read on this call, compared by value since main reads those dicts live and callers both replace and edit them in place. Never key on an object's address the snapshot does not hold, since Python reuses addresses. `Snapshots` hands out one snapshot per generation and config and shares a row cache across generations, keyed by row object identity, so a row no write replaced is lowered once. A host that edits a row object in place calls `Snapshots::forget_rows`. The snapshot reads rows live from the host table, it never copies the table

- `TableRow` and `LiveTable` are the whole host interface: exact key, folded key through the store, and a row object with its fields and `registered_fields`
- `KnownNames` is the two lists only Python has, every provider name and the model names `get_model_info` renames

`pricing.rs` is the glue `litellm-cost` prices through. `Snapshot::pricing_rows` fetches the rows one query may read (each candidate under the model name main priced it with, plus rows by key) into `PricingRows`, the core's `RowSource`, and `with_per_call` adds a row the caller passed for that call alone. `LoweredRow` is the core's `LoweredRow`, and its `RowFacts` (mode, max tokens, how main's `float()` reads `provider_specific_entry`, the raw keys the auth checks read, whether `guardrail_cost_per_unit` validates as main's guardrail pricing) are read off the fields when the row is lowered
- `batch_parity.json` is main's `batch_cost_calculator` answer for deployment rows, captured once. The `batch_parity` tests in `pricing.rs` lower each row here and price it through the core, so they cover the key grammar and the batch rules together. Recapture it only when main's batch pricing changes on purpose
- A row is lowered with `litellm_provider` replaced by the billed provider, as `NativePrices` does

## Copied from main, fix later

Each line is main's behavior kept on purpose. Fixing one is a separate behavior change

- A deployment's price lives in at least seven places that disagree (`costmap/family_config.md` I1 to I7). Only the row the router registered under the deployment id, as stored, is exposed (`RowProvenance::Registered`, I3). Chat bills a direction the deployment did not declare at 0, while batch, the failure estimate and the model group display carry the published rate (I4, `deployment_pricing_model_info` in `litellm_logging.py`) and `per_call_model_info` (I7, `per_call_pricing.py`) rebuilds the row from the catalog whenever any per-call price is set. The I4 and I7 views land with their first callers
- A served model wins over a dated or `@version` request row: Vertex `claude-opus-4-5@20251101` bills `vertex_ai/claude-opus-4-5` when the response names `claude-opus-4-5-20251101` (`_served_or_priced_requested_model`, `cost_calculator.py`)
- A model only a capability rule knows bills from the rule, which prices nothing, so main charges 0 (`get_model_info_helper` generalization). Here it is `Deferral::Generalization`
- A Bedrock Mantle row without token or tier prices bills 0 instead of trying the next candidate (the `else` branch of `cost_per_token`)
- `stripped_model_name` keeps a Mantle region path, `us-gov-west-1/anthropic.claude-x` (`get_potential_model_names`)
- An edit of a row object in place is never seen until the host forgets rows. Main sees it after the next version move, and only where its own caches reread (`VersionCached`)
- A region path is any `<letters-digits-dashes>/` segment (`is_bedrock_region`), where Python checks a fixed region list (`get_all_bedrock_regions`)
- A geo or speed multiplier below 1 is dropped, since `Multiplier` refuses it, where Python applies any number

## Prices

The price vocabulary lives in `litellm-cost`, and only this crate knows JSON key names

A new `N`, image size or resolution in an existing key shape parses with no Rust change. A new unit, token kind, feature or tier needs a variant in `litellm-cost` plus its spelling in `wire/flat_key.rs` and `FLAT_KEY_PATTERN` in `schema.rs`

`Prices::try_from` is the strict lowering the catalog checks use. `billed_prices` lowers the same row for billing and never fails, because a deployment's `model_info` merged over a catalog row can hold what the checks reject. It reads the row the way Python's two pricing paths did

- A row whose token tiers price input bills the whole call from the tier
  - The flat text token prices the tiers replace are dropped, and `BilledPrices::for_prompt(0)` is the only reader of them, since Python picks no tier for an empty prompt
  - A tier's missing cache-read and cache-write rates are that tier's input rate, and its missing reasoning rate is that tier's output rate
  - A token tier starts where the one before it ends, so a ladder written `[0, 128000], [128001, ...]` bills the second tier from token 128001
- Result tiers are left out, since the search calculator prices them
- A modality input token price that does not parse is 0, where a missing one borrows the text rate
- Any other contradiction keeps the price read first

`Adjustments::from` also lowers `provider_specific_entry`. `fast` is the `speed: fast` uplift and every other numeric key is a geo uplift, except `global` and `not_available`, which main never uplifts. Applying them only for Anthropic and Bedrock is the pricing core's rule, not the row's

`wire/blocks.rs` reports a malformed off-peak window or weekday and keeps the well-formed ones, so billing skips only what Python skipped while the catalog checks still reject the row

## Tests

`BUNDLED_REGISTRY` embeds the root `model_prices_and_context_window.json`, so the tests over it fail a JSON change with a bad value or a key no type reads

`Catalog::problems` is the one definition of a mergeable cost map. The bundled test asserts it is empty, and the cost map guard workflow runs it on a pull request's JSON through `cargo run --bin check-registry <path>`
