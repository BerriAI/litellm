# [DEPRECATED] Region-based Routing

:::info

This is deprecated, please use [Tag Based Routing](./tag_routing.md) instead

:::


Route specific customers to eu-only models.

By specifying 'allowed_model_region' for a customer, LiteLLM will filter-out any models in a model group which is not in the allowed region (i.e. 'eu').

[**See Code**](https://github.com/BerriAI/litellm/blob/5eb12e30cc5faa73799ebc7e48fc86ebf449c879/litellm/router.py#L2938)

### 1. Create customer with region-specification

Use the litellm 'end-user' object for this. 

End-users can be tracked / id'ed by passing the 'user' param to litellm in an openai chat completion/embedding call.

```bash
curl -X POST --location 'http://0.0.0.0:4000/end_user/new' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "user_id" : "ishaan-jaff-45",
    "allowed_model_region": "eu", # 👈 SPECIFY ALLOWED REGION='eu'
}'
```

### 2. Add eu models to model-group 

Add eu models to a model group. Use the 'region_name' param to specify the region for each model.

Supported regions are 'eu' and 'us'.

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-35-turbo # 👈 EU azure model
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: os.environ/AZURE_EUROPE_API_KEY
      region_name: "eu"
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      api_key: os.environ/AZURE_API_KEY
      region_name: "us"

router_settings:
  enable_pre_call_checks: true # 👈 IMPORTANT
```

Start the proxy

```yaml
litellm --config /path/to/config.yaml
```

### 3. Test it!

Make a simple chat completions call to the proxy. In the response headers, you should see the returned api base. 

```bash
curl -X POST --location 'http://localhost:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "gpt-3.5-turbo", 
    "messages": [
        {
        "role": "user",
        "content": "what is the meaning of the universe? 1234"
    }],
    "user": "ishaan-jaff-45" # 👈 USER ID
}
'
```

Expected API Base in response headers 

```
x-litellm-api-base: "https://my-endpoint-europe-berri-992.openai.azure.com/"
x-litellm-model-region: "eu" # 👈 CONFIRMS REGION-BASED ROUTING WORKED
```

### FAQ 

**What happens if there are no available models for that region?**

Since the router filters out models not in the specified region, it will return back as an error to the user, if no models in that region are available. 
