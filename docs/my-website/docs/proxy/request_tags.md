# Request Tags for Spend Tracking

Add tags to model deployments to track spend by environment, AWS account, or any custom label.

Tags appear in the `request_tags` field of LiteLLM spend logs.

## Config Setup

Set tags on model deployments in `config.yaml`:

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-prod
      api_key: os.environ/AZURE_PROD_API_KEY
      api_base: https://prod.openai.azure.com/
      tags: ["AWS_IAM_PROD"]  # 👈 Tag for production

  - model_name: gpt-4-dev
    litellm_params:
      model: azure/gpt-4-dev
      api_key: os.environ/AZURE_DEV_API_KEY
      api_base: https://dev.openai.azure.com/
      tags: ["AWS_IAM_DEV"]  # 👈 Tag for development
```

## Make Request

Requests just specify the model - tags are automatically applied:

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Spend Logs

The tag from the model config appears in `LiteLLM_SpendLogs`:

```json
{
  "request_id": "chatcmpl-abc123",
  "request_tags": ["AWS_IAM_PROD"],
  "spend": 0.002,
  "model": "gpt-4"
}
```

## Related

- [Spend Tracking Overview](cost_tracking.md)
- [Tag Budgets](tag_budgets.md) - Set budget limits per tag
