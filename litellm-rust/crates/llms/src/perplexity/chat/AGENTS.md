# rules

- Perplexity through the OpenAI SDK URL rule at `https://api.perplexity.ai`, with `X-Pplx-Integration: litellm`
- Citations become `url_citation` annotations on the first choice, search counts reach `prompt_tokens_details`
- Shared flow and response handling: `../../openai/chat/compatible.rs`

# references

- https://docs.perplexity.ai/api-reference/chat-completions-post
- litellm/llms/perplexity/chat/transformation.py
