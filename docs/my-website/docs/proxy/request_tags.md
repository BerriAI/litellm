import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Request Tags for Spend Tracking

Add tags to your LLM requests to track spend by environment, AWS account, team, or any custom label.

Tags appear in the `request_tags` field of LiteLLM spend logs.

## Set Default Tags in Config

Set default tags for all API keys generated through the proxy:

```yaml title="config.yaml"
litellm_settings:
  default_key_generate_params:
    metadata:
      tags: ["AWS_IAM_PROD", "us-east-1"]
```

All keys created via `/key/generate` will automatically include these tags.

## Send Tags in Request

Pass tags in the request body:

<Tabs>
<TabItem value="curl" label="cURL">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {
      "tags": ["AWS_IAM_PROD", "us-east-1"]
    }
  }'
```

</TabItem>
<TabItem value="header" label="Header">

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -H 'x-litellm-tags: AWS_IAM_PROD,us-east-1' \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

</TabItem>
</Tabs>

## Where Tags Appear

Tags are stored in `LiteLLM_SpendLogs` under `request_tags`:

```json
{
  "request_id": "chatcmpl-abc123",
  "request_tags": ["AWS_IAM_PROD", "us-east-1"],
  "spend": 0.002,
  "model": "gpt-4"
}
```

## Related

- [Spend Tracking Overview](cost_tracking.md)
- [Tag Budgets](tag_budgets.md) - Set budget limits per tag
