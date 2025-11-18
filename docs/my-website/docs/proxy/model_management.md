import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Management
Add new models + Get model info without restarting proxy.

## In Config.yaml 

```yaml
model_list:
  - model_name: text-davinci-003
    litellm_params: 
      model: "text-completion-openai/text-davinci-003"
    model_info: 
      metadata: "here's additional metadata on the model" # returned via GET /model/info
```

## Get Model Information - `/model/info`

Retrieve detailed information about each model listed in the `/model/info` endpoint, including descriptions from the `config.yaml` file, and additional model info (e.g. max tokens, cost per input token, etc.) pulled from the model_info you set and the [litellm model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). Sensitive details like API keys are excluded for security purposes.

:::tip Sync Model Data
Keep your model pricing data up to date by [syncing models from GitHub](../sync_models_github.md).
:::

<Tabs
  defaultValue="curl"
  values={[
    { label: 'cURL', value: 'curl', },
  ]}>
  <TabItem value="curl">

```bash
curl -X GET "http://0.0.0.0:4000/model/info" \
     -H "accept: application/json" \
```
  </TabItem>
</Tabs>

## Add a New Model

Add a new model to the proxy via the `/model/new` API, to add models without restarting the proxy.

<Tabs>
<TabItem value="API">

```bash
curl -X POST "http://0.0.0.0:4000/model/new" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{ "model_name": "azure-gpt-turbo", "litellm_params": {"model": "azure/gpt-3.5-turbo", "api_key": "os.environ/AZURE_API_KEY", "api_base": "my-azure-api-base"} }'
```
</TabItem>
<TabItem value="Yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo ### RECEIVED MODEL NAME ### `openai.chat.completions.create(model="gpt-3.5-turbo",...)`
    litellm_params: # all params accepted by litellm.completion() - https://github.com/BerriAI/litellm/blob/9b46ec05b02d36d6e4fb5c32321e51e7f56e4a6e/litellm/types/router.py#L297
      model: azure/gpt-turbo-small-eu ### MODEL NAME sent to `litellm.completion()` ###
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_EU" # does os.getenv("AZURE_API_KEY_EU")
      rpm: 6      # [OPTIONAL] Rate limit for this deployment: in requests per minute (rpm)
    model_info: 
      my_custom_key: my_custom_value # additional model metadata
```

</TabItem>
</Tabs>


### Model Parameters Structure

When adding a new model, your JSON payload should conform to the following structure:

- `model_name`: The name of the new model (required).
- `litellm_params`: A dictionary containing parameters specific to the Litellm setup (required).
- `model_info`: An optional dictionary to provide additional information about the model.

Here's an example of how to structure your `ModelParams`:

```json
{
  "model_name": "my_awesome_model",
  "litellm_params": {
    "some_parameter": "some_value",
    "another_parameter": "another_value"
  },
  "model_info": {
    "author": "Your Name",
    "version": "1.0",
    "description": "A brief description of the model."
  }
}
```
---

Keep in mind that as both endpoints are in [BETA], you may need to visit the associated GitHub issues linked in the API descriptions to check for updates or provide feedback:

- Get Model Information: [Issue #933](https://github.com/BerriAI/litellm/issues/933)
- Add a New Model: [Issue #964](https://github.com/BerriAI/litellm/issues/964)

Feedback on the beta endpoints is valuable and helps improve the API for all users.


## Add Additional Model Information 

If you want the ability to add a display name, description, and labels for models, just use `model_info:` 

```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "gpt-4"
      api_key: "os.environ/OPENAI_API_KEY"
    model_info: # 👈 KEY CHANGE
      my_custom_key: "my_custom_value"
```

### Usage

1. Add additional information to model 

```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "gpt-4"
      api_key: "os.environ/OPENAI_API_KEY"
    model_info: # 👈 KEY CHANGE
      my_custom_key: "my_custom_value"
```

2. Call with `/model/info` 

Use a key with access to the model `gpt-4`.

```bash
curl -L -X GET 'http://0.0.0.0:4000/v1/model/info' \
-H 'Authorization: Bearer LITELLM_KEY' \
```

3. **Expected Response**

Returned `model_info = Your custom model_info + (if exists) LITELLM MODEL INFO`


[**How LiteLLM Model Info is found**](https://github.com/BerriAI/litellm/blob/9b46ec05b02d36d6e4fb5c32321e51e7f56e4a6e/litellm/proxy/proxy_server.py#L7460) 

[Tell us how this can be improved!](https://github.com/BerriAI/litellm/issues)

```bash
{
    "data": [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4"
            },
            "model_info": {
                "id": "e889baacd17f591cce4c63639275ba5e8dc60765d6c553e6ee5a504b19e50ddc",
                "db_model": false,
                "my_custom_key": "my_custom_value", # 👈 CUSTOM INFO
                "key": "gpt-4", # 👈 KEY in LiteLLM MODEL INFO/COST MAP - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
                "max_tokens": 4096,
                "max_input_tokens": 8192,
                "max_output_tokens": 4096,
                "input_cost_per_token": 3e-05,
                "input_cost_per_character": null,
                "input_cost_per_token_above_128k_tokens": null,
                "output_cost_per_token": 6e-05,
                "output_cost_per_character": null,
                "output_cost_per_token_above_128k_tokens": null,
                "output_cost_per_character_above_128k_tokens": null,
                "output_vector_size": null,
                "litellm_provider": "openai",
                "mode": "chat"
            }
        },
    ]
}
```
