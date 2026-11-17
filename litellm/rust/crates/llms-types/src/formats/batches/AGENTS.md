# rules

- The batch contract follows OpenAI's Batch object, which is LiteLLM's normalized batch shape for every provider
- Providers with their own batch API map into it in `llms/src/<provider>/batches/` (Anthropic Message Batches)
- Statuses and counts a provider lacks are derived or defaulted in the provider transformation, not added here

# references

- https://developers.openai.com/api/reference/resources/batches
