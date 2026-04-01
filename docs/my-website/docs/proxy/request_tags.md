import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Request Tags for Spend Tracking

Add tags to model deployments to track spend by environment, AWS account, or any custom label.

Tags appear in the `request_tags` field of LiteLLM spend logs.

:::info Requirements
Virtual Keys & a database should be set up. See [Virtual Keys Setup](./virtual_keys.md).
:::

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

### Option 1: Use Config Tags (Automatic)

Requests just specify the model - tags are automatically applied from config:

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Option 2: Use `x-litellm-tags` Header

Pass tags dynamically via the `x-litellm-tags` header as a comma-separated string:

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -H 'x-litellm-tags: team-api,production,us-east-1' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Format: Comma-separated string (spaces are automatically trimmed): `"tag1,tag2,tag3"`

### Option 3: Use Request Body `tags`

Pass tags directly in the request body. Both formats are supported:

<Tabs>
<TabItem value="direct" label="Direct tags Field">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "tags": ["team-api", "production", "us-east-1"]
  }'
```

</TabItem>

<TabItem value="metadata" label="Metadata Nested">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {
      "tags": ["team-api", "production", "us-east-1"]
    }
  }'
```

</TabItem>
</Tabs>

The `tags` field must be an array of strings.

:::info
When tags are provided via header or request body, they override any tags configured in the model deployment. If both header and body tags are provided, body tags take precedence.
:::

## Set Tags on Keys or Teams

You can also set default tags at the API key or team level:

<Tabs>
<TabItem value="key" label="Set on Key">

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "metadata": {
      "tags": ["customer-acme", "tier-premium"]
    }
  }'
```

</TabItem>
<TabItem value="team" label="Set on Team">

```bash
curl -L -X POST 'http://0.0.0.0:4000/team/new' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "metadata": {
      "tags": ["team-engineering", "department-ai"]
    }
  }'
```

</TabItem>
</Tabs>

## Advanced: Custom Header Tracking

Track spend using any custom header by adding it to your config:

```yaml
litellm_settings:
  extra_spend_tag_headers:
    - "x-custom-header"
    - "x-customer-id"
```

**Disable User-Agent tracking:**

```yaml
litellm_settings:
  disable_add_user_agent_to_request_tags: true
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

- [Spend Tracking Overview](cost_tracking.md) - Complete tutorial on tracking spend with tags
- [Tag Budgets](tag_budgets.md) - Set budget limits per tag
- [Virtual Keys Setup](virtual_keys.md) - Required for tag tracking
