# rules

- Owns the Messages adapter contract, its execution inputs such as `MessagesTransformContext`, and provider-independent machinery
- A context carries only inputs the contract needs, not every provider's settings
- Shared normalization implements LiteLLM's provider-independent Messages input contract
- Metadata filtering, tool-ID rewriting, web-search replay, beta selection, thinking translation and thinking budgets are provider policy. Adapters opt into shared provider helpers explicitly
- Restrictions common to Claude hosts are still provider policy, not format rules
