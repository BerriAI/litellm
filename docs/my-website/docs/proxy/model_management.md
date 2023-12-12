# Model Management
Add new models + Get model info without restarting proxy.

## Get Model Information

Retrieve detailed information about each model listed in the `/models` endpoint, including descriptions from the `config.yaml` file, and additional model info (e.g. max tokens, cost per input token, etc.) pulled the model_info you set and the litellm model cost map. Sensitive details like API keys are excluded for security purposes.

<Tabs
  defaultValue="curl"
  values={[
    { label: 'cURL', value: 'curl', },
  ]}>
  <TabItem value="curl">

```bash
curl -X GET "http://0.0.0.0:8000/model/info" \
     -H "accept: application/json" \
```
  </TabItem>
</Tabs>

## Add a New Model

Add a new model to the list in the `config.yaml` by providing the model parameters. This allows you to update the model list without restarting the proxy.

<Tabs
  defaultValue="curl"
  values={[
    { label: 'cURL', value: 'curl', },
  ]}>
  <TabItem value="curl">

```bash
curl -X POST "http://0.0.0.0:8000/model/new" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{ "model_name": "azure-gpt-turbo", "litellm_params": {"model": "azure/gpt-3.5-turbo", "api_key": "os.environ/AZURE_API_KEY", "api_base": "my-azure-api-base"} }'
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