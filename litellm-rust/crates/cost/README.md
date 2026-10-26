# litellm-cost

- `calculate(&prices, &usage, &when)` consumes the same `Prices` that `litellm-model-catalog` produces. There is no separate pricing specification or compilation step, and no runtime caller is migrated here
- `Usage::quantities` contains disjoint billable quantities keyed by `Meter`, including text, audio, image, reasoning, cache reads, writes by TTL, characters, seconds, images and queries. Text counts exclude the quantities assigned to sub-meters. Provider response normalization stays with the caller
  - `prompt_tokens` is the total prompt count, including cache and modalities, used only for selecting a prompt band. It is not billed a second time
  - `inclusive_threshold` chooses `>=` instead of `>` at prompt-band boundaries
  - `search_results` selects a query result band. Query quantity remains the number of queries. Video-second quantity selects a duration interval
- `When` supplies the service tier, channel and billing timestamp. `price::When` remains the catalog key for a tier and band
- `Priced::Known` contains a `Breakdown`. `Priced::Unpriced` lists every used meter without a price and retains the priced lines and their partial total. Zero usage needs no price. An explicit zero price remains a known zero-cost line
- Every line records its billed meter, quantity, unit price and amount. When a sub-meter falls back, `fallback_from` identifies the original usage meter and `meter` identifies the price source
  - Audio, image and video tokens fall back to text on the same side, and reasoning falls back to text output
  - Cache reads fall back to text input. Long-TTL writes fall back through the default write price to text input
  - Missing non-token prices never fall back to token prices
- Online lookup preserves R1's `(tier, band)`, `(standard, band)`, `(tier, base)`, `(standard, base)` order before sub-meter fallback
  - Off-peak overrides only declared rates during matching UTC windows. Windows can wrap midnight and restrict weekdays in an IANA timezone
  - Batch prices select bands per meter. Explicit batch prices win, missing input sub-meters use the batch input rate, and missing batch token rates can use half the online base rate
  - Decisions use their own channel and never silently borrow online prices
- `Breakdown::input()` sums non-output lines and `output()` sums output lines. `total` sums all lines. Prices and amounts remain money newtypes with `f64` internally
- Regional adjustments remain separate from prices. The caller selects and applies `Adjustments` when needed
- Run `cargo run -p litellm-cost --example charge` for a normalized usage example. The existing Python reference TSV is still tested unchanged. Broader fixture generation and full Python pricing parity are deferred
