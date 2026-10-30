# litellm-cost

This crate holds the price data model that `litellm-model-catalog` lowers each catalog entry into, plus the usage, call context and priced outcome types that `calculate` will compute over. There is no `calculate` until the token pricing PR lands. `UnitPrice` is USD per unit and `Usd` is an amount, and only the operations that make sense between them exist, so a price is never added to an amount. `Prices::band` and `Prices::rate` resolve a token rate the way Python's `_get_token_base_cost` does
