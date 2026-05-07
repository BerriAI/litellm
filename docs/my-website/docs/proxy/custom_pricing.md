import Image from '@theme/IdealImage';

# Custom LLM Pricing

## Overview

LiteLLM provides flexible cost tracking and pricing customization for all LLM providers:

- **Custom Pricing** - Override default model costs or set pricing for custom models
- **Cost Per Token** - Track costs based on input/output tokens (most common)
- **Cost Per Second** - Track costs based on runtime (e.g., Sagemaker)
- **Zero-Cost Models** - Bypass budget checks for free/on-premises models by setting costs to 0
- **[Provider Discounts](./provider_discounts.md)** - Apply percentage-based discounts to specific providers
- **[Provider Margins](./provider_margins.md)** - Add fees/margins to LLM costs for internal billing
- **Base Model Mapping** - Ensure accurate cost tracking for Azure deployments

By default, the response cost is accessible in the logging object via `kwargs["response_cost"]` on success (sync + async). [**Learn More**](../observability/custom_callback.md)

:::info

LiteLLM already has pricing for 100+ models in our [model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). 

:::

## Cost Per Second (e.g. Sagemaker)

#### Usage with LiteLLM Proxy Server

**Step 1: Add pricing to config.yaml**
```yaml
model_list:
  - model_name: sagemaker-completion-model
    litellm_params:
      model: sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4
    model_info:
      input_cost_per_second: 0.000420
  - model_name: sagemaker-embedding-model
    litellm_params:
      model: sagemaker/berri-benchmarking-gpt-j-6b-fp16
    model_info:
      input_cost_per_second: 0.000420 
```

**Step 2: Start proxy**

```bash
litellm /path/to/config.yaml
```

**Step 3: View Spend Logs**

<Image img={require('../../img/spend_logs_table.png')} />

## Cost Per Token (e.g. Azure)

#### Usage with LiteLLM Proxy Server

```yaml
model_list:
  - model_name: azure-model
    litellm_params:
      model: azure/<your_deployment_name>
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: os.environ/AZURE_API_VERSION
    model_info:
      input_cost_per_token: 0.000421 # 👈 ONLY to track cost per token
      output_cost_per_token: 0.000520 # 👈 ONLY to track cost per token
```

## Database-Stored Models (`/model/new` + `STORE_MODEL_IN_DB=True`)

:::warning Different placement for DB-stored models

When models are added at runtime via [`POST /model/new`](./model_management.md#add-a-new-model) and persisted to Postgres (with `general_settings.store_model_in_db: true`), custom-pricing fields must be placed in **`litellm_params`**, not in `model_info`.

The `model_info` JSONB column is **not** mirrored into per-request metadata for DB-loaded deployments, so `_response_cost_calculator` never sees pricing placed there and falls back to the static catalog. The result is `response_cost: 0.0` in spend logs even though the value is visible in `GET /model/info`.

This differs from the config.yaml flow (above), where pricing in `model_info` works because the YAML loader wires `model_info` into request metadata at startup.

:::

### Working `/model/new` example

```bash
curl -X POST "http://0.0.0.0:4000/model/new" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model_name": "my-custom-llm",
      "litellm_params": {
        "model": "openai/some-openai-compatible-model",
        "api_base": "https://api.example.com/v1",
        "api_key": "os.environ/SOME_API_KEY",
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003
      },
      "model_info": {
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
        "max_input_tokens": 128000,
        "max_output_tokens": 4096
      }
    }'
```

The duplicate copy under `model_info` is **optional** — it is only used for `GET /model/info` UI/inspection responses. The copy under `litellm_params` is what actually drives cost calculation in spend logs.

### Updating prices on an existing DB-stored model

For partial updates, use `PATCH /model/{model_id}/update`. Place the cost fields under `litellm_params`:

```bash
curl -X PATCH "http://0.0.0.0:4000/model/<model_id>/update" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model_info": {"id": "<model_id>"},
      "litellm_params": {
        "input_cost_per_token": 0.0000014,
        "output_cost_per_token": 0.0000044
      }
    }'
```

Updated pricing applies to **subsequent** requests immediately — `PATCH` calls `clear_cache()`, which reloads DB-stored deployments and re-invokes `Router.register_model()` with the new pricing. No proxy restart is required.

:::note Prefer `PATCH /model/{model_id}/update` over the legacy `POST /model/update`

The legacy `POST /model/update` endpoint **does** persist cost-field changes to the database, but it does **not** call `clear_cache()`. The router's in-memory state is not refreshed, so the new prices remain inactive until the next proxy restart. The per-id `PATCH /model/{model_id}/update` form (shown above) updates the database AND hot-reloads the router in the same request, so prices take effect immediately.

:::

### Why custom_pricing only flows from `litellm_params` for DB models

The pricing-detection helper `use_custom_pricing_for_model` (in `litellm/litellm_core_utils/litellm_logging.py`) inspects:

1. `litellm_params.<cost_field>` directly, **or**
2. `litellm_params.metadata.model_info.<cost_field>` / `litellm_params.litellm_metadata.model_info.<cost_field>`.

Config-yaml deployments populate (2) when the YAML loader builds Deployment objects. DB-loaded deployments do not — `model_info` from the JSONB column is exposed via `/model/info` for UI but is not propagated into request metadata. As a result, only (1) — pricing in `litellm_params` — is observed by the cost calculator for DB models.

Tracking issue: [BerriAI/litellm#15135](https://github.com/BerriAI/litellm/issues/15135) (closed as "not planned"; documenting the canonical placement here so users do not hit silent `$0.00` spend tracking).

## Override Model Cost Map

You can override [our model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) with your own custom pricing for a mapped model.

Just add a `model_info` key to your model in the config, and override the desired keys.

Example: Override Anthropic's model cost map for the `prod/claude-3-5-sonnet-20241022` model.

```yaml
model_list:
  - model_name: "prod/claude-3-5-sonnet-20241022"
    litellm_params:
      model: "anthropic/claude-3-5-sonnet-20241022"
      api_key: os.environ/ANTHROPIC_PROD_API_KEY
    model_info:
      input_cost_per_token: 0.000006
      output_cost_per_token: 0.00003
      cache_creation_input_token_cost: 0.0000075
      cache_read_input_token_cost: 0.0000006
```

### Additional Cost Keys

There are other keys you can use to specify costs for different scenarios and modalities:

- `input_cost_per_token_above_200k_tokens` - Cost for input tokens when context exceeds 200k tokens
- `output_cost_per_token_above_200k_tokens` - Cost for output tokens when context exceeds 200k tokens  
- `cache_creation_input_token_cost_above_200k_tokens` - Cache creation cost for large contexts
- `cache_read_input_token_cost_above_200k_token` - Cache read cost for large contexts
- `input_cost_per_image` - Cost per image in multimodal requests
- `output_cost_per_reasoning_token` - Cost for reasoning tokens (e.g., OpenAI o1 models)
- `input_cost_per_audio_token` - Cost for audio input tokens
- `output_cost_per_audio_token` - Cost for audio output tokens
- `input_cost_per_video_per_second` - Cost per second of video input
- `input_cost_per_video_per_second_above_128k_tokens` - Video cost for large contexts
- `input_cost_per_character` - Character-based pricing for some providers
- `input_cost_per_token_priority` / `output_cost_per_token_priority` - Priority/PayGo pricing (Vertex AI Gemini, Bedrock)
- `input_cost_per_token_flex` / `output_cost_per_token_flex` - Batch/flex pricing

These keys evolve based on how new models handle multimodality. The latest version can be found at [https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

### Service Tier / PayGo Pricing (Vertex AI, Bedrock)

For providers that support multiple pricing tiers (e.g., Vertex AI PayGo, Bedrock service tiers), LiteLLM automatically applies the correct cost based on the response:

- **Vertex AI Gemini**: Uses `usageMetadata.trafficType` (`ON_DEMAND_PRIORITY` → priority, `FLEX`/`BATCH` → flex). See [Vertex AI - PayGo / Priority Cost Tracking](../providers/vertex.md#paygo--priority-cost-tracking).
- **Bedrock**: Uses `serviceTier` from the response. See [Bedrock - Usage - Service Tier](../providers/bedrock.md#usage---service-tier).

## Zero-Cost Models (Bypass Budget Checks)

**Use Case**: You have on-premises or free models that should be accessible even when users exceed their budget limits.

**Solution** ✅: Set both `input_cost_per_token` and `output_cost_per_token` to `0` (explicitly) to bypass all budget checks for that model.

:::info

When a model is configured with zero cost, LiteLLM will automatically skip ALL budget checks (user, team, team member, end-user, organization, and global proxy budget) for requests to that model.

**Important**: Both costs must be **explicitly set to 0**. If costs are `null` or undefined, the model will be treated as having cost and budget checks will apply.

:::

### Configuration Example

```yaml
model_list:
  # On-premises model - free to use
  - model_name: on-prem-llama
    litellm_params:
      model: ollama/llama3
      api_base: http://localhost:11434
    model_info:
      input_cost_per_token: 0   # 👈 Explicitly set to 0
      output_cost_per_token: 0  # 👈 Explicitly set to 0
  
  # Paid cloud model - budget checks apply
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
    # No model_info - uses default pricing from cost map
```

### Behavior

With the above configuration:

- **User over budget** → Can still use `on-prem-llama` ✅, but blocked from `gpt-4` ❌
- **Team over budget** → Can still use `on-prem-llama` ✅, but blocked from `gpt-4` ❌
- **End-user over budget** → Can still use `on-prem-llama` ✅, but blocked from `gpt-4` ❌

This ensures your free/on-premises models remain accessible regardless of budget constraints, while paid models are still properly governed.

## Set 'base_model' for Cost Tracking (e.g. Azure deployments)

**Problem**: Azure returns `gpt-4` in the response when `azure/gpt-4-1106-preview` is used. This leads to inaccurate cost tracking

**Solution** ✅ :  Set `base_model` on your config so litellm uses the correct model for calculating azure cost

Get the base model name from [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

Example config with `base_model`
```yaml
model_list:
  - model_name: azure-gpt-3.5
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
    model_info:
      base_model: azure/gpt-4-1106-preview
```

### OpenAI Models with Dated Versions

`base_model` is also useful when OpenAI returns a dated model name in the response that differs from your configured model name.

**Example**: You configure custom pricing for `gpt-4o-mini-audio-preview`, but OpenAI returns `gpt-4o-mini-audio-preview-2024-12-17` in the response. Since LiteLLM uses the response model name for pricing lookup, your custom pricing won't be applied.

**Solution** ✅: Set `base_model` to the key you want LiteLLM to use for pricing lookup.

```yaml
model_list:
  - model_name: my-audio-model
    litellm_params:
      model: openai/gpt-4o-mini-audio-preview
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      base_model: gpt-4o-mini-audio-preview  # 👈 Used for pricing lookup
      input_cost_per_token: 0.0000006
      output_cost_per_token: 0.0000024
      input_cost_per_audio_token: 0.00001
      output_cost_per_audio_token: 0.00002
```


## Debugging 

If you're custom pricing is not being used or you're seeing errors, please check the following:

1. Run the proxy with `LITELLM_LOG="DEBUG"` or the `--detailed_debug` cli flag

```bash
litellm --config /path/to/config.yaml --detailed_debug
```

2. Check logs for this line: 

```
LiteLLM:DEBUG: utils.py:263 - litellm.acompletion
```

3. Check if 'input_cost_per_token' and 'output_cost_per_token' are top-level keys in the acompletion function. 

```bash
acompletion(
  ...,
  input_cost_per_token: my-custom-price, 
  output_cost_per_token: my-custom-price,
)
```

If these keys are not present, LiteLLM will not use your custom pricing. 

If the problem persists, please file an issue on [GitHub](https://github.com/BerriAI/litellm/issues). 
