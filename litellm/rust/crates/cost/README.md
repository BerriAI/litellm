# litellm-cost

This crate holds the price data model that `litellm-model-catalog` lowers each catalog entry into, and the pricing over it. `Prices::quote` settles the service tier, prompt size and off-peak window of a call, and `Quote::rate` is the one rule for what a meter costs, including which meter a missing price borrows from. `calculate` multiplies usage over a quote, sums each side in the order Python summed it and applies the uplifts the call names. `UnitPrice` is USD per unit and `Usd` is an amount, and only the operations that make sense between them exist, so a price is never added to an amount

Three entry points answer queries over a `RowSource`, which Model Catalog implements over its per-call snapshot

- `price` takes a `PriceQuery` tagged by kind and answers a `Receipt` or says why the call is unpriced. The `tokens` kind prices Claude on Anthropic, Bedrock, Vertex AI and Azure AI with main's speed, geo, regional and web search rules, discount and margin. Every other kind, and any provider or case not ported yet, answers `NotPorted` so Python prices it
- `bound` is the legacy budget hold, `estimate_request_max_cost` and `estimate_request_input_cost` ported as they are
- `rates` gives a row's rates, the raw key facts the auth predicates read and one priced, free or unpriced status

A batch or decisions call, or one that carries custom pricing in its `CallContext`, gets a quote with no rates, since Python still prices those
