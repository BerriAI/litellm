# External model offerings

Set `general_settings.model_offerings_path` in the normal proxy configuration and keep its `model_list` empty. Disable `store_model_in_db`; the database may still serve authentication and usage tracking. Existing proxy configurations behave as before when the path is absent

```yaml
model_list: []
general_settings:
  model_offerings_path: /app/offerings/offerings.yaml
  store_model_in_db: false
```

Mount a host directory into Docker, for example `./offerings:/app/offerings:ro`. Keep `offerings.yaml` inside it. Replace the file atomically within that directory so the container sees the new inode. A bind mount of an individual file does not reliably see atomic replacement

```yaml
version: 1
config_poll_seconds: 5
inventory_poll_seconds: 300
providers:
  openrouter:
    provider: openrouter
    api_key: os.environ/OPENROUTER_API_KEY
  vercel:
    provider: vercel_ai_gateway
    api_key: os.environ/VERCEL_API_KEY
  local:
    provider: openai
    api_base: http://inference-server:8000/v1
    api_key: os.environ/INFERENCE_API_KEY
  subscribed:
    provider: chatgpt
offerings:
  - model_name: selected
    source: auto
    provider: openrouter
    upstream_model: organization/model
    enabled: true
  - model_name: handwritten
    source: manual
    provider: openrouter
    upstream_model: organization/another-model
    model_info:
      mode: chat
      context_window: 262144
      supports_function_calling: true
      request_defaults:
        output_token_budget: 8192
        output_token_budget_by_reasoning_effort:
          low: 65536
          high: 65536
          xhigh: 65536
          max: 131072
```

These example token values illustrate operator configuration, not a claim about the example supplier models. Request defaults express chosen budgets and do not populate independent output capability limits

`source` belongs to each offering. AUTO entries require membership in the latest successful full inventory for their connection. Missing entries disappear from model listings and return a useful unavailable error before forwarding. A returning entry becomes available again. AUTO metadata combines the verified catalog with downloaded supplier fields and then explicit overrides, in increasing priority, and freezes the result in the serving snapshot. MANUAL entries use only handwritten definitions and are independent of supplier inventory membership. Both sources accept shared `model_info` overrides. Wildcards and duplicate public aliases are rejected

Startup validates the complete file and awaits initial inventory requests. AUTO entries remain unavailable when no successful inventory exists; MANUAL entries remain usable. A successful empty inventory means absence. HTTP errors, authentication failures, timeouts and malformed responses retain the last successful inventory for unchanged credentials. Changing connection credentials starts with unknown availability and forces a fresh inventory. ChatGPT uses its existing native OAuth account and cannot initiate device login from discovery or request handling

The file is checked every five seconds by default, with a configurable range of 1 to 300 seconds. Inventories refresh every 300 seconds by default, with a range of 1 to 86400 seconds. Changing that interval triggers a fresh inventory request and resets its deadline. Invalid file edits retain the whole last-valid serving configuration, including credentials, and log only the error kind. Unselected inventory entries never become public routes. Caller connection overrides and runtime model-management mutations are rejected in external mode; edit the external file instead

Each HTTP request pins one immutable serving snapshot through the complete ASGI response lifetime, including streaming. Publishing a new snapshot changes subsequent requests while requests already running finish on their previous native Router. Snapshot creation isolates deployment indexes and metadata caches while reusing the initialized Router's bounded client, usage and routing resources. It does not register fresh callbacks or close shared HTTP clients on each reload. The polling task stops at proxy shutdown after requests drain. Inventory refreshes reacquire the dedicated shared HTTP client through the native cache, whose eviction lifecycle owns client closure


Native subscription inventory uses the server-owned backend from `CHATGPT_API_BASE` / `OPENAI_CHATGPT_API_BASE`, or the provider default. An explicit connection URL must match that backend; per-model destinations cannot receive saved server OAuth credentials. Discovery never starts device login. Supplier-specific inventory behavior lives in the shared `litellm/llms/model_inventory.py` dispatcher.

Connection identities use HMAC with the existing private server salt/master key. With that key configured, deployment IDs remain stable across restarts and change when supplier credentials or native account identity changes. No additional configured secret is required. Unconfigured standalone/development mode uses a private process key and does not promise ID stability across restart. Upgrading from earlier unkeyed offering IDs changes IDs once: encrypted reasoning tied to a removed ID may be discarded on replay or model switch, while readable history and model limits/defaults remain available. Native affinity callbacks use the request-pinned serving view during reloads.

Caller `extra_body.model` overrides are rejected before supplier forwarding; harmless extra body fields remain supported. Structured `reasoning_effort` objects such as `{effort: high, summary: auto}` select the same configured output budget as the corresponding effort string.

## Provider endpoint policy

An offering can own a typed `provider_policy`. Currently OpenRouter chat offerings support an exact endpoint allowlist with fallbacks disabled. The policy belongs to the offering, not the connection, and never accepts credentials or arbitrary request-body options. The supplier must match the policy provider and `model_info.mode` must explicitly be `chat`

```yaml
offerings:
  - model_name: selected-endpoint
    source: auto
    provider: openrouter
    upstream_model: organization/model
    enabled: false
    provider_policy:
      provider: openrouter
      only: [supplier/exact-endpoint]
      allow_fallbacks: false
    model_info:
      mode: chat
      context_window: 32000
      max_output_tokens: 4000
```

Append this entry to the external file's existing `offerings`, using its configured OpenRouter connection name. The example limits are synthetic overrides. Omit them to discover bounds automatically. `only` contains provider slugs, not URLs. Base slugs include their variants, and variant slugs include their regions, following OpenRouter's matching contract. Service-tier endpoints require an explicit tier slug. Multiple slugs intentionally allow any matching endpoint; `allow_fallbacks` must remain false

The gateway merges its policy into harmless caller `extra_body` fields and verifies it again at deployment dispatch. Caller `provider`, `models`, `route`, `service_tier`, fallback controls, parameter-dropping/allowlist controls, nested body overrides and mock responses are rejected for policy-controlled offerings, even through aliases. Tier overrides could change the eligible endpoint set and invalidate its advertised guarantees; they are rejected rather than silently reduced or rerouted. Unpinned tier controls remain unchanged. Caller deployment selectors are rejected throughout external mode. Policy-controlled requests disable LiteLLM model-group fallbacks as well as OpenRouter provider fallback. Unavailable endpoints fail instead of serving a different offering. Retries may still contact the same pinned endpoint

Policy-controlled requests disable response-cache reads and writes, overriding caller cache settings. Existing response-cache keys do not reliably separate provider endpoints; cached output from another endpoint must not satisfy a pinned request

Chat Completions, native Responses and translated Anthropic Messages preserve the policy, including streaming. Other APIs are rejected for policy-controlled offerings. Batch execution remains unavailable throughout external mode. Offerings without policy retain existing provider-routing behavior, apart from the deployment-selector restriction

AUTO still checks model membership. Pinned AUTO offerings additionally discover only their selected models' endpoint inventories, using the configured connection, bounded native HTTP client and credential-scoped cache. Aggregate model limits and bundled metadata never substitute for pinned endpoint limits. Context, independent input and output bounds are the conservative minimum across all eligible endpoints, and a field unknown on any eligible endpoint is omitted. Input capacity is not inferred by subtracting output from context

Reasoning choices and defaults come from OpenRouter's model-level reasoning contract and are advertised only when every eligible endpoint declares the effort control. An endpoint's more restrictive declared choices narrow that set. This describes OpenRouter's accepted gateway options, not an independently discovered backend-native effort enum. An explicit empty choice list remains empty. Explicit null expands only to OpenRouter's documented gateway efforts, while omission does not declare choices. Mandatory reasoning excludes the disable value, and a disabled-by-default model does not claim an active default effort. Boolean reasoning support alone never supplies choices. Endpoint capability guarantees are combined conservatively, without using status, uptime or performance to change routing or availability

Every explicit `model_info` field remains authoritative. Existing handwritten context/output limits continue overriding discovery until the operator removes them. MANUAL offerings remain entirely handwritten. Requests without an effort leave the supplier default unchanged; explicit gateway effort values, including `max` and `xhigh`, are forwarded without backend-flag approximation on chat and translated Messages paths

Successful empty endpoint inventories clear derived endpoint metadata. Failed refreshes retain last-known endpoint metadata only for the same connection identity; an initial failure leaves unknown fields omitted, never replaced with another supplier's limits. Endpoint discovery does not disable an otherwise selected model or broaden its policy. Metadata refreshes publish atomically, and requests already running retain their previous snapshot. Policy edits still change the affected offering's private deployment ID

The endpoint and reasoning provenance follows OpenRouter's [endpoint API](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model), [per-model reasoning options](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options) and [provider slug matching](https://openrouter.ai/docs/guides/routing/provider-selection#base-slug-matching), checked October 15, 2026

Native provider batch creation and uploads with purpose `batch` are rejected in external offering mode. Native batch records can select their own supplier models, which would bypass the configured offering catalog. Existing proxy configurations without external offerings retain native batch support.

The optional bundled cost-map JSON-schema extension is deferred: its trusted-base guard validates PR artifacts with the base generator. The runtime metadata/configuration contract validates separate total/input/output limits and defaults independently, and unknown optional bundled catalog properties remain forward compatible.
