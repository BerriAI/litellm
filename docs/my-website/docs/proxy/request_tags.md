import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Request Tags for Spend Tracking

Add tags to your LLM requests to track spend by environment, AWS account, team, or any custom label.

Tags appear in the `request_tags` field of LiteLLM spend logs, making it easy to filter and analyze costs.

## Quick Start

Add tags via the `metadata.tags` field in your request:

<Tabs>
<TabItem value="python" label="Python SDK">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "metadata": {
            "tags": ["AWS_IAM_PROD", "us-east-1"]
        }
    }
)
```

</TabItem>
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
</Tabs>

## Where Tags Appear

Tags are stored in the `LiteLLM_SpendLogs` table under `request_tags`:

```json
{
  "request_id": "chatcmpl-abc123",
  "request_tags": ["AWS_IAM_PROD", "us-east-1"],
  "spend": 0.002,
  "model": "gpt-4",
  ...
}
```

## Common Use Cases

| Tag Example | Purpose |
|-------------|---------|
| `AWS_IAM_PROD` | Track requests from production AWS account |
| `AWS_IAM_DEV` | Track requests from development AWS account |
| `team-backend` | Attribute costs to backend team |
| `project-chatbot` | Track spend for a specific project |

## Set Default Tags on API Keys

Set tags at the API key level so all requests automatically inherit them:

```bash
curl -X POST 'http://0.0.0.0:4000/key/generate' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "metadata": {
      "tags": ["AWS_IAM_PROD", "team-backend"]
    }
  }'
```

## Related

- [Spend Tracking Overview](cost_tracking.md)
- [Tag Budgets](tag_budgets.md) - Set budget limits per tag
- [Tag Routing](tag_routing.md) - Route requests based on tags
