# structure

- The shared Messages provider adapter contract (`BaseMessagesConfig`), its execution inputs such as `MessagesTransformContext`, and provider-independent transformation and streaming machinery

# boundaries

- Public request, response, content-block and event schemas belong in `litellm-llms-types::formats::messages`
- Call orchestration belongs in `inference-messages/src`
- Metadata filtering, tool-ID rewriting, web-search replay policy, beta selection and thinking translation belong in `llms/src/<provider>/messages`

# invariants

- Nothing here imports a provider implementation or embeds provider policy in trait defaults, normalization or context defaults
- Shared normalization implements only LiteLLM's provider-independent Messages input contract
- A context carries only the inputs the shared adapter contract needs, not every provider's settings
- Adapters opt into shared helpers explicitly, so a helper never runs for a provider that did not ask for it

# gotchas

- Thinking-budget choices and model-specific restrictions are not format rules just because several providers host Claude

# references

- `litellm/llms/base_llm/anthropic_messages/transformation.py`
