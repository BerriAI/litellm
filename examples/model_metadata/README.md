# Gateway model discovery

LiteLLM advertises model limits and capabilities in its normal `/v1/models` response and detailed `/model/info` response

The shared metadata contract distinguishes `context_window`, the total combined input and generated output limit, from `max_input_tokens` and `max_output_tokens`. Unknown values are omitted. LiteLLM does not add the two maxima to derive context, or use a context value as an output limit

Configure the models you want to expose in `model_list`. LiteLLM retrieves metadata for those models from supported upstream catalogs at startup, then refreshes every five minutes. OpenAI-compatible servers can publish the shared fields, `context_length`, or `max_model_len`. OpenRouter and Vercel catalogs are normalized from their provider-specific fields. ChatGPT uses the active account's authenticated native model catalog without starting a device login

Upstream metadata overrides LiteLLM's bundled model catalog. Explicit `model_info` values override upstream metadata. A server that publishes only model IDs still works, and its missing limits can be configured manually in LiteLLM

```yaml
model_list:
  - model_name: shared-model
    litellm_params:
      model: openai/provider-model
      api_base: https://inference.example/v1
      api_key: os.environ/PROVIDER_API_KEY
    model_info:
      context_window: 120000
      max_input_tokens: 110000
      max_output_tokens: 20000
      supports_function_calling: true
      supports_parallel_function_calling: false
      supports_reasoning: true
      reasoning_effort_levels: [low, high, xhigh]
      default_reasoning_effort: high
      supported_endpoints: [/v1/chat/completions]
      supported_modalities: [text, image]
      supported_output_modalities: [text]
```

These example limits describe independent constraints. With 20000 output tokens reserved, this example allows at most 100000 input tokens because the total context limit also applies

Reasoning levels are the exact values the configured route accepts. The list is not restricted to a fixed set of vendor names, and a default outside a known list is omitted. Explicit `false` capabilities and empty lists are preserved

ChatGPT's native catalog supplies account-specific reasoning choices, defaults, input modalities, and parallel tool support. Its client policy windows remain separate `codex_context_window` and `codex_max_context_window` diagnostics in detailed model information. They are not advertised as physical total context or input limits. Missing canonical limits require verified catalog values or a manual `model_info` definition

Native Codex UI reasoning aliases are normalized to API effort values. `ultra` is not advertised as a wire effort, and `persistent` maps to `disabled`

For a model group with an explicit total context contract, discovery advertises the smallest known input, output, and total limit across its deployments. A limit is omitted if any deployment does not declare it. Capabilities require agreement across deployments, supported lists are intersected, and a default must agree across deployments. Existing input/output aggregation is preserved for legacy groups without a total context field

Only configured models are exposed. Fetching a supplier's catalog does not add its other models to the downstream listing. Set `general_settings.disable_model_info_refresh: true` to disable upstream catalog polling

Configured output budgets are request defaults, independent of supplier limits. Set `model_info.request_defaults` to advertise and apply the same policy centrally

```yaml
model_info:
  context_window: 262144
  request_defaults:
    output_token_budget: 8192
    output_token_budget_by_reasoning_effort:
      low: 65536
      high: 65536
      xhigh: 65536
      max: 131072
```

The route chooses a configured budget from the effective request or deployment reasoning effort when the caller omits all explicit output budget parameters. An unmatched effort uses `output_token_budget`. Chat Completions and Anthropic Messages receive `max_tokens`, and Responses receives `max_output_tokens`. Caller-provided `max_tokens`, `max_completion_tokens`, or `max_output_tokens` takes precedence. The policy object is not sent to the supplier

Discovery advertises `request_defaults` only when every deployment has the same policy. These chosen budgets do not populate `max_output_tokens`, establish a supplier maximum, or change the total context window. Clients can reserve the selected budget during compaction while the supplier's independent output limit remains unknown
