# Native /v1/messages passthrough

By default, when you call LiteLLM's unified `/v1/messages` endpoint against a deployment whose provider does not have a native Anthropic Messages config (for example `openai` or `hosted_vllm`), LiteLLM translates the inbound Anthropic request down to `/v1/chat/completions` (or the Responses API for `openai`). That translation drops Anthropic-only features such as `cache_control` and `thinking`

Some OpenAI-compatible model servers (self-hosted vLLM, DeepSeek's Anthropic-compatible endpoint, and similar) also expose the native Anthropic `/v1/messages` API. For those deployments you can tell LiteLLM to forward the raw Anthropic payload straight through to `{api_base}/v1/messages` with no translation, so `cache_control`, `thinking`, `tools`, and the rest of the Anthropic shape are preserved end to end

## Opting in

Set `supported_endpoints` under the deployment's `model_info` and include `"/v1/messages"`. When that entry is present, `/v1/messages` requests are passed through natively; `/v1/chat/completions` requests to the same deployment keep their existing native OpenAI behavior, so a single deployment can serve both surfaces

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: openai/some-model
      api_base: https://host/v1
      api_key: os.environ/SOME_KEY
    model_info:
      supported_endpoints: ["/v1/chat/completions", "/v1/messages"]
```

The opt-in is per deployment and provider-agnostic. Without `"/v1/messages"` in `supported_endpoints`, behavior is unchanged and the request is still translated

## How the request is forwarded

The Anthropic payload (`model`, `messages`, `system`, `max_tokens`, `stop_sequences`, `temperature`, `top_p`, `top_k`, `tools`, `tool_choice`, `thinking`, `metadata`, `stream`, and any `cache_control` blocks inside `messages` / `system`) is sent essentially unchanged. `max_tokens` is required, as it is for the Anthropic API

The downstream URL is built from `api_base` as `{api_base}/v1/messages`. An `api_base` already ending in `/v1` or `/v1/messages` is handled without doubling the path, so `https://host/v1` becomes `https://host/v1/messages` and `https://api.deepseek.com/anthropic` becomes `https://api.deepseek.com/anthropic/v1/messages`

Authorization is sent as `Authorization: Bearer {api_key}`, which OpenAI-compatible servers and DeepSeek's Anthropic endpoint accept. `content-type: application/json` and a default `anthropic-version: 2023-06-01` are added when the caller did not already supply them; headers you pass through are never overwritten
