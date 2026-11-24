The price vocabulary (meter, band, channel, `When`, `Billing`) is explained in [../model-catalog/AGENTS.md](../model-catalog/AGENTS.md#concepts), the only crate that maps it to JSON keys

## Layout

- `price`, `bound` and `rates` are the three entry points, and `RowSource` (`row.rs`) is the only way they reach a row. `candidates::PricingRows` implements it over the rows a catalog's `PricingTable` hands over, so this crate never reads the cost map, never merges rows and holds no provider list beyond the pricing rules it ports
- `engine.rs` hands each `PriceQuery` kind to its handler module. A kind or case without a handler answers `Unpriced::NotPorted`, and only then may the caller price it the old way
- `tokens.rs` and `tokens/` port the token path of `completion_cost`: `rule.rs` for what each provider branch of `cost_per_token` adds, `breakdown.rs` for the values main stores beside the total, `custom.rs` for `custom_cost_per_token`
- `adjust.rs` ports the discount and margin config. Rules keep the order the config wrote them in
- `candidates/` is completion_cost's candidate walk for the `/v1/messages` Claude providers (`CostProvider`): which names to try in which order, deployment rows by id, which provider to bill as, and when main would price another way (`AfterRows::Deferred` with its `RustRefusal`, so Python keeps pricing it). Rows before a deferral are still tried first, because main tries them first. It reads a catalog only through `CandidateTable` and `PricingTable`, which model-catalog implements, so this crate never depends on model-catalog. `candidates::priced` is the token-call entry point every host calls with a `TokenCall`: it walks the candidates, fetches their rows into `PricingRows` (the `RowSource`) and prices the query. A row the caller passes for one call is lowered by the host and added with `with_per_call`. Its parity tests run in model-catalog (`resolve/tests/candidate_walk.rs`), the only place both are visible

## Parity fixtures

`fixtures/*.json` were captured once from main's Python with `LITELLM_LOCAL_MODEL_COST_MAP=True`. Each case holds its inputs (the row fields it read, the billed usage, the config) and main's outputs, and the `parity` test module of each kind asserts Rust equals them to the last bit. Never replay Python from a test. When a fixture must change, recapture it from main and say why in the commit

## Copied from main, fix later

- Anthropic drops both the speed and the geo uplift when either `provider_specific_entry` value it reads is not a number (`except Exception: pass` in `llms/anthropic/cost_calculation.py` `cost_per_token`)
- The stored breakdown rates carry every residency, FedRAMP, regional, geo and speed uplift the call names whatever the provider, while each total carries only its provider's (`llm_cost_calc/utils.py` `_cost_map_billed_rates`)
- Bedrock applies geo only to a row it looked up by name, never to a registered or caller-given row, and never applies speed (`cost_calculator.py` `cost_per_token` bedrock branch, `llms/bedrock/cost_calculation.py`)
- Bedrock raises on a geo value that is not a number and moves on to the next candidate, since nothing catches it (`llms/bedrock/cost_calculation.py`)
- Bedrock bills one web search at the requested context size when the usage reports zero search requests (`tool_call_cost_tracking.py` `_billable_web_search_calls`)
- A row without an input or output rate bills that direction at 0.0 instead of moving on to the next candidate (`utils.py` `_get_model_info_helper` defaults), listed in `Receipt::defaulted_to_zero`
- With `requested_model_info_override` the stored breakdown and the web search fee read the catalog row the candidate names, not the override that priced the total (`cost_calculator.py` `completion_cost`, `tool_call_cost_tracking.py` `_resolve_model_info`)
- Prompt-caching savings leave out the geo and speed uplifts the total carries (`llm_cost_calc/utils.py` `calculate_prompt_caching_savings`)
- The legacy budget hold reads base rates, the tier a prompt picks, and for a counted Gemini body the `_above_<n>_tokens` prices of the row and its group's `deployment_pricing`. Cache-write rates, uplifts, discount and margin never reach it, and a router group's reasoning and image rates are dropped before it. The adapter (`proxy/spend_tracking/budget_reservation.py` `_bound`) reads which output caps reach the provider and how many outputs it returns per model, so `bound` reads only counts and flags off the request

## Not copied from main

- Main's stored breakdown raises on an unreadable speed or geo value, which fails the candidate when a logging object is attached. Rust leaves those uplifts out of the breakdown rates instead
- A call whose web search price main would read off the served model's row answers `NotPorted`
- The breakdown bills unattributed writes and 5m/30m writes as one `CacheTtl::Default` count. Main multiplies each count by the rate separately, so the two can differ in the last bit when a call reports both
