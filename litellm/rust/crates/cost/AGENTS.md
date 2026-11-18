The price vocabulary (meter, band, channel, `When`, `Billing`) is explained in [../model-catalog/AGENTS.md](../model-catalog/AGENTS.md#concepts), the only crate that maps it to JSON keys

## Layout

- `price`, `bound` and `rates` are the three entry points, and `RowSource` (`row.rs`) is the only way they reach a row. Model Catalog implements it over its per-call snapshot, so this crate never reads the cost map, never merges rows and holds no provider list beyond the pricing rules it ports
- `engine.rs` hands each `PriceQuery` kind to its handler module. A kind or case without a handler answers `Unpriced::NotPorted`, and only then may the caller price it the old way
- `tokens.rs` and `tokens/` port the token path of `completion_cost`: `rule.rs` for what each provider branch of `cost_per_token` adds, `breakdown.rs` for the values main stores beside the total, `custom.rs` for `custom_cost_per_token`
- `adjust.rs` ports the discount and margin config. Rules keep the order the config wrote them in
- Each per-unit kind has its own module (`rerank.rs`, `vector_store.rs`, `ocr.rs`, `speech.rs`, `transcription.rs`, `search.rs`, `video.rs`, `image.rs`), which picks the meter and quantity main's calculator billed. `units.rs` holds what they share: the rows a header names, `PriceSource::Rows` (the deployment's own price tables, read ahead of the catalog row in the order the kind's calculator reads them), billing one row through `calculate`, and the receipt with the query's discount and margin settled
  - Where main billed a call nothing on purpose (no row, no rate, no units) the core answers `Unpriced`, never 0.0, and the adapter maps it to each calculator's legacy return
- `batch.rs` prices one line of a batch output file from the row its caller resolved, reading each rate the way main's batch helpers read its key. `price_many` prices many queries against one `RowSource`, so the bridge lowers each row of a batch output file once and crosses into the core once per file
- The fee kinds (`a2a.rs`, `mcp.rs`, `guardrail.rs`) price what the caller configured (`PriceSource::PerQuery`, `FlatPerToken`, an MCP server's tool prices, Azure Prompt Shield's price per thousand records) or the guardrail rows the header lists. A hook-set `response_cost` that beats them stays in the adapter. A call billed as several separately priced units lists each in `Receipt::parts`, where a unit no price covers stays `Unpriced` and never reads as free

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
- The legacy budget hold reads flat rates only, so bands, cache-write rates, uplifts, discount and margin never reach it, and a router group's reasoning and image rates are dropped before it (`proxy/spend_tracking/budget_reservation.py` `_max_cost_for_cost_info`)
- A streamed `/v1/messages` call never bills the service tier its usage reports, because the assembled stream response drops `usage.service_tier`, so only the requested tier prices it (`inference-messages/src/pricing.rs` copies this from `stream_chunk_builder`)
- The per-event stream cost injection prices each `message_delta`'s own usage alone, without the `message_start` input and cache counts, the request's speed or the stream's thinking (`common_request_processing.py` `_inject_cost_into_usage_dict`, copied in `inference-messages/src/pricing.rs`)

- A rerank row priced only per token bills 0 unless its provider's config bills tokens (Jina AI, Scaleway, Voyage), as Cohere v5, Azure AI Cohere v5 and Together AI rows are billed by search units they have no rate for (`calculate_rerank_cost` of `BaseRerankConfig`)
- An OCR annotation page with no annotation rate of its own bills at the page rate, also in a batch, where the page rate may come from the deployment while the catalog prices annotations (`ocr_cost`, `ocr_batch_cost`)
- A Vertex AI Lyria generation bills the bundled cost map's rate when the live row has none, so a live table can never unprice it (`get_vertex_ai_lyria_generation_cost`, the adapter still passes both rows)
- A speech or transcription call priced per token, and a transcription whose usage reports tokens on a row that declares token rates, are `NotPorted` and bill through main's generic token path, which bills a row's synthesized zero token rates
- Parallel AI bills each result past the ten a request includes at a fixed rate kept in `search.rs`, since no cost map row lists `cost_per_additional_result` (`PARALLEL_AI_ADDITIONAL_RESULT_COST`)
- A search row priced by `max_results_range` tiers falls back to the last tier when the requested results fall below the first range. A row listing any `tiered_pricing` answers `NotPorted`, even beside a flat query rate main never reads then, since billing lowering leaves result tiers out, and the adapter keeps main's tier pick
- A deployment's video table replaces the catalog row whole, so a deployment that prices only input images bills no video, and one that prices only per second never reads the catalog's clip table (`video_generation_cost`)
- A video length of exactly half a second rounds to the even clip length, so 4.5s reads the `4s` cell (Python's `round` in `video_clip_cost`)
- The default image calculator bills `input_cost_per_image` for returned images when a row lists it, ahead of `output_cost_per_image`, and reads the deployment's table whole before the catalog row, so a deployment pixel rate beats a catalog per-image rate (`default_image_cost_calculator`)
- A per-image provider bills a returned image the deployment or catalog prices at 0, and one neither prices at 0.0 rather than failing (`recraft`, `aiml`, `cometapi`, `runwayml` cost calculators)

- A Bedrock guardrail counter its pricing row does not list is unknown in the per-counter rollup but adds 0 to the billed total (`guardrail_cost_total`), and Azure Prompt Shield bills nothing on the free tier whatever its usage

- A batch line's image and video input tokens read the flat batch rate whatever the prompt's size, as main reads them with no prompt tokens (`_batch_prompt_cost`), and its image output tokens read only flat rates
- A batch modality rate crosses the bands either the online or the batch price list names, and a crossed band the batch list does not price reads the flat batch rate instead of the online band (`_optional_batch_rate`)
- A batch row with no batch input rate bills every cache write at half the base write rate, whatever lifetime the usage reports (`_half_price_prompt_cost`)

## Not copied from main
- `video_generation_cost` read a catalog row through `get_model_info`, which drops every `output_cost_per_second_<resolution>` key but six. The core reads any resolution the row lists. No shipped row lists another

- Main's stored breakdown raises on an unreadable speed or geo value, which fails the candidate when a logging object is attached. Rust leaves those uplifts out of the breakdown rates instead
- A call whose web search price main would read off the served model's row answers `NotPorted`
- A batch line whose cached and written tokens exceed its prompt tokens bills no uncached input, where main's half-price path billed a negative amount (`_half_price_prompt_cost`)
- A negative or non-numeric configured agent, MCP tool or Prompt Shield price fails the call's pricing instead of billing a negative amount, and a guardrail row whose `guardrail_cost_per_unit` lists a negative rate prices none of its counters, where main billed the negative rate. An MCP tool listed with a null price fails too, where main returned `None`
- The breakdown bills unattributed writes and 5m/30m writes as one `CacheTtl::Default` count. Main multiplies each count by the rate separately, so the two can differ in the last bit when a call reports both
