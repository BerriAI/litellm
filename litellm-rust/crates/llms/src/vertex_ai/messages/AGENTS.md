# structure

- Vertex AI's native Messages adapter for Claude partner models (`vertex_ai/<claude model>`): endpoint, GCP auth policy, beta selection and Vertex request shaping

# boundaries

- Token minting and refresh belong in the `auth-gcp` crate. This folder only sends the resulting bearer token
- Responses and streams are Anthropic wire format and decode through `anthropic/messages` with no Vertex rewriting

# invariants

- Only `vertex_ai` models whose lowercased name contains `claude` use this adapter. Every other Vertex model goes through the chat-completion adapter, so do not widen the gate
- The location is untrusted input that ends up in a hostname, so validate it before building the URL: `global`, or a lowercase token of letters, digits and hyphens
- Resolve project, location and credentials without consuming them from the caller's params
- Copy headers before changing them so shared deployment headers stay untouched
- `model` is removed from the body because Vertex takes it only from the URL
- The URL and the environment never disagree: Python fixes the URL during environment validation, and here project and location resolve from the same settings through one `VertexTarget::resolve` every time

# gotchas

- Forwarded `Authorization` is always replaced by explicit/configured credentials or token minting, matching Python refresh behavior; forwarded-bearer-only callers need another credential source
- The model catalog's `supported_regions` overrides the location: an unset location takes the first entry and an unsupported one is rerouted to it
- A location without a hyphen, such as `us` or `eu`, is a multi-region geography with its own `rep` host, not a malformed region
- Tool search tools add Vertex's own tool search beta, not the first-party one
- `scope` is stripped from every `cache_control` and billing metadata system blocks are always stripped
- `output_config.effort` is dropped for models that do not accept effort on Vertex

# known gaps

- Project and location come only from `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `VERTEX_LOCATION`, because core does not pass `vertex_project`, `vertex_location` or `vertex_credentials` from `litellm_params` to Messages configs yet
- A project is required up front. Python can take it from the credentials, but that lookup is async and the URL is built synchronously, so a missing project is a configuration error here
- The model catalog's `supported_regions` override is not applied, since no catalog lookup reaches this adapter, so the location defaults to `us-central1`
- Tokens are minted through a process-wide `VertexAuth` in this module, not the `AuthServices.gcp` cache that OCR uses, because `AuthScheme` has no Vertex variant that `resolve_auth` could route to it
- A request-controlled `api_base` is not rejected before a minted token is sent to it, unlike OCR, because Messages configs do not see where `api_base` came from
- `scope` is stripped from system, message and top-level `cache_control`, but not from tool definitions

# references

## python

- `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/experimental_pass_through/transformation.py` (`VertexAIPartnerModelsAnthropicMessagesConfig`)
- `litellm/llms/vertex_ai/vertex_llm_base.py` (`VertexBase`)
- `litellm/llms/vertex_ai/common_utils.py` (`get_vertex_base_url`)
- `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/output_params_utils.py` (`sanitize_vertex_anthropic_output_params`)
- `litellm/utils.py` (`ProviderConfigManager._get_provider_anthropic_messages_config_cached`, adapter selection)

## docs

- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/claude/use-claude
- https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai.md
- https://platform.claude.com/docs/en/api/http/messages/create
