# litellm-cost

This crate holds the price data model that `litellm-model-catalog` lowers each catalog entry into, and `calculate`, which prices the token, character, image, second and query meters of one online call over it. `Prices::quote` settles the service tier, prompt size and off-peak window of a call, and `Quote::rate` is the one rule for what a meter costs, including which meter a missing price borrows from. `calculate` multiplies usage over a quote, sums each side in the order Python summed it and applies the residency, FedRAMP and regional endpoint uplifts. `UnitPrice` is USD per unit and `Usd` is an amount, and only the operations that make sense between them exist, so a price is never added to an amount

The batch channel, per-call custom pricing and the Anthropic speed and geo uplifts are still priced in Python
