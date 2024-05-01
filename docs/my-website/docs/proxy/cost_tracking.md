# Cost Tracking - Azure

Set base model for cost tracking azure image-gen call

## Image Generation 

```yaml
model_list: 
  - model_name: dall-e-3
    litellm_params:
        model: azure/dall-e-3-test
        api_version: 2023-06-01-preview
        api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
        api_key: os.environ/AZURE_API_KEY
        base_model: dall-e-3 # 👈 set dall-e-3 as base model
    model_info:
        mode: image_generation
```

## Chat Completions / Embeddings

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