# structure

- Azure's Messages adapter for Claude on Microsoft Foundry, reached by `azure_ai/<claude model>` on the Messages route

# boundaries

- Anthropic payload shaping is reused from `anthropic/messages` because Foundry hosts the same backend, and Azure-only rewrites stay here
- Limiting this adapter to Claude deployments is core selection's job, not this folder's

# invariants

- `secret_names` lists every env name the adapter reads
- A forwarded `x-api-key` or a non-blank `Authorization: Bearer` (an Entra ID token) is the credential, and the adapter adds no key on top of it
- The Azure key goes in `x-api-key`, not Azure's usual `api-key`
- The adapter removes `cache_control.scope` from system and message blocks
- Scope removal and system folding stay idempotent; the current adapter folds all system-role turns into top-level `system`

# gotchas

- Foundry accepts only `type` and `ttl` in `cache_control`, so `scope` is stripped even though the first-party API accepts it
- Foundry rejects some features the first-party API accepts (code execution, Files API, newer web search and web fetch versions). That is upstream policy and the adapter does not filter them
- Foundry omits the `anthropic-ratelimit-*` response headers and adds `apim-request-id` next to `request-id`
- The chat sibling `litellm/llms/azure_ai/anthropic/transformation.py` is not part of this route

# known gaps

- Core maps every `azure_ai` model to this adapter, while Python uses it only when the lowercased model name contains `claude` and sends the rest through the chat-completions bridge
- Python also reads `litellm.api_key`, `litellm.azure_key` and `AZURE_OPENAI_API_KEY` and mints an Entra ID token from tenant, client and secret params. Rust does none of these

# references

## python

- `litellm/llms/azure_ai/anthropic/messages_transformation.py` (`AzureAnthropicMessagesConfig`)
- `litellm/llms/azure/common_utils.py` (`BaseAzureLLM._base_validate_azure_environment`, key lookup)
- `ProviderConfigManager._get_provider_anthropic_messages_config_cached` in `litellm/utils.py` (Claude-only selection)

## docs

- https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry
- https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry.md
- https://platform.claude.com/llms.txt
- https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-claude
- https://platform.claude.com/docs/en/api/http/messages/create
