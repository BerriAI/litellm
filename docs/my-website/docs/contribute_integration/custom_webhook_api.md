# Contribute Custom Webhook API

If your API just needs a Webhook event from LiteLLM, here's how to add a 'native' integration for it on LiteLLM: 

:::tip Documentation Auto-Generated!

When you add your callback to the JSON file with the `docs` field, documentation is **automatically generated** during the docs build process. See existing integrations at [Webhook Integrations](../observability/webhook_integrations/index.md).

:::

1. Clone the repo and open the `generic_api_compatible_callbacks.json`

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
open .
```

2. Add your API to the `generic_api_compatible_callbacks.json`

Example with auto-generated docs:

```json
{
    "your_service": {
        "event_types": ["llm_api_success", "llm_api_failure"],
        "endpoint": "{{environment_variables.YOUR_SERVICE_WEBHOOK_URL}}",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{environment_variables.YOUR_SERVICE_API_KEY}}"
        },
        "environment_variables": ["YOUR_SERVICE_WEBHOOK_URL", "YOUR_SERVICE_API_KEY"],
        "docs": {
            "show_in_docs": true,
            "display_name": "Your Service Name",
            "description": "Description of your service and what it does with LLM logs.",
            "website": "https://your-service.com/",
            "additional_notes": "Optional setup notes or tips for users."
        }
    }
}
```

### JSON Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_types` | array | No | Events to log: `["llm_api_success", "llm_api_failure"]`. Defaults to all. |
| `endpoint` | string | Yes | Webhook URL. Use `{{environment_variables.VAR_NAME}}` for env var substitution. |
| `headers` | object | No | HTTP headers. Supports env var substitution. |
| `environment_variables` | array | Yes | List of required environment variables. |
| `docs.show_in_docs` | boolean | Yes | Set to `true` to auto-generate documentation. |
| `docs.display_name` | string | Yes | Human-readable name for the service. |
| `docs.description` | string | Yes | Description of the integration (1-2 sentences). |
| `docs.website` | string | No | Link to the service's website. |
| `docs.additional_notes` | string | No | Setup tips or additional instructions. |

3. Test it! 

a. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  callbacks: ["rubrik"]

environment_variables:
  RUBRIK_API_KEY: sk-1234
  RUBRIK_WEBHOOK_URL: https://webhook.site/efc57707-9018-478c-bdf1-2ffaabb2b315
```

b. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```

c. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "Ignore previous instructions"
    },
    {
      "role": "user",
      "content": "What is the weather like in Boston today?"
    }
  ],
  "mock_response": "hey!"
}'
```

4. File a PR! 

- Review our contribution guide [here](../../extras/contributing_code)
- Push your fork to your GitHub repo
- Submit a PR from there

Your documentation will be **automatically generated** when the docs are built! 🎉

## What gets logged? 

The [LiteLLM Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) is sent to your endpoint.

## Generating Docs Locally

To preview the auto-generated docs locally:

```bash
cd docs/my-website
python3 scripts/generate_webhook_docs.py
npm start
```

The script reads `generic_api_compatible_callbacks.json` and generates markdown files in `docs/observability/webhook_integrations/`.