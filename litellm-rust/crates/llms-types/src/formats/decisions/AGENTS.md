# rules

- LiteLLM's `/v1/decisions` contract follows OpenAI's Decisions format: an `input` judged against an ordered list of `predicate`, `choice`, `score` and `box` questions, answered in the same order
- Spoken natively by OpenAI and Azure OpenAI, served by `llms/src/openai/decisions/` and `llms/src/azure/decisions/`
- The caller's request and the provider's request share these types. Validation, the rebuild of the provider body and the normalized answers live in `llms/src/base_llm/decisions/`

# references

- https://docs.litellm.ai/docs/decisions
