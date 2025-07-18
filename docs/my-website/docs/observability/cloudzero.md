import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CloudZero - Unit Cost Analytics

[CloudZero](https://www.cloudzero.com/) provides cloud cost intelligence and unit economics tracking for engineering teams.

## What this does

Logs cost data from LiteLLM to CloudZero's Unit Cost Analytics API, enabling you to:
- Track LLM costs alongside your other cloud costs
- Attribute costs to specific users, teams, and organizations
- Create unit economics metrics (e.g., cost per customer, cost per feature)
- Analyze cost trends and optimize spending

## Quick Start
Use just 1 line of code to instantly log your LLM costs **across all providers** to CloudZero

Get your CloudZero API Key from your CloudZero account settings.

```python
litellm.callbacks = ["cloudzero"] # logs cost + usage of successful calls to CloudZero
```

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
import os

# CloudZero configuration
os.environ["CLOUDZERO_API_KEY"] = "your-api-key-here"
os.environ["CLOUDZERO_METRIC_NAME"] = "llm-cost" # The metric name in CloudZero
os.environ["CLOUDZERO_API_BASE"] = "https://api.cloudzero.com" # Optional, defaults to https://api.cloudzero.com

# LLM API Keys
os.environ['OPENAI_API_KEY'] = ""

# set cloudzero as a callback, litellm will send cost data to CloudZero
litellm.success_callback = ["cloudzero"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add to your litellm config.yaml
```yaml
litellm_settings:
  callbacks: ["cloudzero"]
```

2. Set the required environment variables
```bash
export CLOUDZERO_API_KEY="your-api-key-here"
export CLOUDZERO_METRIC_NAME="llm-cost"
```

3. Start the proxy
```bash
litellm --config config.yaml
```

</TabItem>
</Tabs>

## Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `CLOUDZERO_API_KEY` | Yes | Your CloudZero API key | - |
| `CLOUDZERO_METRIC_NAME` | Yes | The metric name to use in CloudZero | - |
| `CLOUDZERO_API_BASE` | No | CloudZero API base URL | `https://api.cloudzero.com` |

## What gets logged

The CloudZero integration sends the following data:
- **value**: The cost of the LLM call in USD
- **timestamp**: When the call was made (ISO 8601 format)
- **filters**: Additional metadata for cost attribution
  - `model`: The LLM model used
  - `user_id`: The LiteLLM user ID (if available)
  - `team_id`: The LiteLLM team ID (if available)
  - `org_id`: The LiteLLM organization ID (if available)
  - `end_user_id`: The end user ID (if available)

## Example CloudZero Telemetry Payload

```json
{
  "records": [
    {
      "value": 0.0015,
      "timestamp": "2024-01-15T10:30:00Z",
      "filters": {
        "model": "gpt-3.5-turbo",
        "user_id": "user-123",
        "team_id": "team-456",
        "org_id": "org-789",
        "end_user_id": "customer-999"
      }
    }
  ]
}
```

## Using CloudZero Unit Cost Analytics

Once data is flowing to CloudZero, you can:

1. **View costs in CloudZero Explorer**: Navigate to the CloudZero platform to see your LLM costs alongside other cloud costs
2. **Create unit metrics**: Define custom metrics like "LLM cost per customer" or "LLM cost per API call"
3. **Set up anomaly detection**: Get alerts when LLM costs spike unexpectedly
4. **Attribute costs**: Use the filters to break down costs by team, user, or customer

## Troubleshooting

If data isn't appearing in CloudZero:

1. Verify your API key is correct
2. Check that the metric name matches what's configured in CloudZero
3. Ensure the CloudZero API is accessible from your network
4. Look for error messages in the LiteLLM logs with `LITELLM_LOG=DEBUG`

## Support

- CloudZero Documentation: https://docs.cloudzero.com/docs/cloudzero
- CloudZero Unit Cost Analytics: https://docs.cloudzero.com/docs/cloudzero/cost-allocation/unit-cost-analytics
- CloudZero API Reference: https://docs.cloudzero.com/reference
- LiteLLM Issues: https://github.com/BerriAI/litellm/issues