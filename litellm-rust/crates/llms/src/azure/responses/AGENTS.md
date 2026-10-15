# rules

- Build the URL like Python's `BaseAzureLLM._get_base_azure_url`: the `preview`, `latest` and `v1` API versions go to `/openai/v1/responses`, and dated versions go to `/openai/responses`. `AZURE_DEFAULT_RESPONSES_API_VERSION` sets the default, which is `preview`
- Authenticate with the `api-key` header from `api_key`, `AZURE_OPENAI_API_KEY` or `AZURE_API_KEY`, and keep a forwarded `api-key` header. Azure AD token flows stay on Python
- Lift Chat Completions-style `function` tools to the top level, and drop `status` from replayed reasoning and message items
- No native WebSocket mode

# references

- https://learn.microsoft.com/azure/ai-foundry/openai/how-to/responses
