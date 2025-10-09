import Image from '@theme/IdealImage';

# Custom LLM Pricing

Use this to register custom pricing for models. 

There's 2 ways to track cost: 
- cost per token
- cost per second

By default, the response cost is accessible in the logging object via `kwargs["response_cost"]` on success (sync + async). [**Learn More**](../observability/custom_callback.md)

:::info

LiteLLM already has pricing for any model in our [model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). 

:::

## Cost Per Second (e.g. Sagemaker)

### Usage with LiteLLM Proxy Server

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

### Usage with LiteLLM Proxy Server

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

These keys evolve based on how new models handle multimodality. The latest version can be found at [https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

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
