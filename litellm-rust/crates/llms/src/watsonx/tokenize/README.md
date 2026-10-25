# Watsonx tokenization

When `include` is omitted, the shared tokenization route returns the representations Watsonx provides. Watsonx returns token strings and leaves numeric token IDs as `null`. An explicit empty `include` requests count-only output. Explicit representation lists override the default, and requesting unsupported token IDs follows the route's existing unsupported-representation error contract
