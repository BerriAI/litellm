import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sumo Logic

Send LiteLLM logs to Sumo Logic for observability, monitoring, and analysis.

Sumo Logic is a cloud-native machine data analytics platform that provides real-time insights into your applications and infrastructure.
https://www.sumologic.com/

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

1. Create a Sumo Logic account at https://www.sumologic.com/
2. Set up an HTTP Logs and Metrics Source in Sumo Logic:
   - Go to **Manage Data** > **Collection** > **Collection**
   - Click **Add Source** next to a Hosted Collector
   - Select **HTTP Logs & Metrics**
   - Copy the generated URL (it contains the authentication token)

For more details, see the [HTTP Logs & Metrics Source](https://www.sumologic.com/help/docs/send-data/hosted-collectors/http-source/logs-metrics/) documentation.

```shell
pip install litellm
```

## Quick Start

Use just 2 lines of code to instantly log your LLM responses to Sumo Logic.

The Sumo Logic HTTP Source URL includes the authentication token, so no separate API key is required.

<Tabs>
<TabItem value="python" label="SDK">

```python
litellm.callbacks = ["sumologic"]
```

```python
import litellm
import os

# Sumo Logic HTTP Source URL (includes auth token)
os.environ["SUMOLOGIC_WEBHOOK_URL"] = "https://collectors.sumologic.com/receiver/v1/http/your-token-here"

# LLM API Keys
os.environ['OPENAI_API_KEY'] = ""

# Set sumologic as a callback
litellm.callbacks = ["sumologic"]

# OpenAI call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - I'm testing Sumo Logic integration"}
  ]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: ["sumologic"]

environment_variables:
  SUMOLOGIC_WEBHOOK_URL: os.environ/SUMOLOGIC_WEBHOOK_URL
```

2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Hey, how are you?"
    }
  ]
}'
```

</TabItem>
</Tabs>

## What Data is Logged?

LiteLLM sends the [Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) to Sumo Logic, which includes:

- **Request details**: Model, messages, parameters
- **Response details**: Completion text, token usage, latency
- **Metadata**: User ID, custom metadata, timestamps
- **Cost tracking**: Response cost based on token usage

Example payload:

```json
{
  "id": "chatcmpl-123",
  "call_type": "litellm.completion",
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "response": {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "Hi there!"
      }
    }]
  },
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  },
  "response_cost": 0.0001,
  "start_time": "2024-01-01T00:00:00",
  "end_time": "2024-01-01T00:00:01"
}
```

## Advanced Configuration

### Batching Settings

Control how LiteLLM batches logs before sending to Sumo Logic:

<Tabs>
<TabItem value="python" label="SDK">

```python
import litellm

os.environ["SUMOLOGIC_WEBHOOK_URL"] = "https://collectors.sumologic.com/receiver/v1/http/your-token"

litellm.callbacks = ["sumologic"]

# Configure batch settings (optional)
# These are inherited from CustomBatchLogger
# Default batch_size: 100
# Default flush_interval: 60 seconds
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml
litellm_settings:
  callbacks: ["sumologic"]

environment_variables:
  SUMOLOGIC_WEBHOOK_URL: os.environ/SUMOLOGIC_WEBHOOK_URL
```

</TabItem>
</Tabs>

### Compressed Data

Sumo Logic supports compressed data (gzip or deflate). LiteLLM automatically handles compression when beneficial.

Benefits:
- Reduced network usage
- Faster message delivery
- Lower data transfer costs

### Query Logs in Sumo Logic

Once logs are flowing to Sumo Logic, you can query them using the Sumo Logic Query Language:

```sql
_sourceCategory=litellm
| json "model", "response_cost", "usage.total_tokens" as model, cost, tokens
| sum(cost) by model
```

Example queries:

**Total cost by model:**
```sql
_sourceCategory=litellm
| json "model", "response_cost" as model, cost
| sum(cost) as total_cost by model
| sort by total_cost desc
```

**Average response time:**
```sql
_sourceCategory=litellm
| json "start_time", "end_time" as start, end
| parse regex field=start "(?<start_ms>\d+)"
| parse regex field=end "(?<end_ms>\d+)"
| (end_ms - start_ms) as response_time_ms
| avg(response_time_ms) as avg_response_time
```

**Requests per user:**
```sql
_sourceCategory=litellm
| json "model_parameters.user" as user
| count by user
```

## Authentication

The Sumo Logic HTTP Source URL includes the authentication token, so you only need to set the `SUMOLOGIC_WEBHOOK_URL` environment variable.

**Security Best Practices:**
- Keep your HTTP Source URL private (it contains the auth token)
- Store it in environment variables or secrets management
- Regenerate the URL if it's compromised (in Sumo Logic UI)
- Use separate HTTP Sources for different environments (dev, staging, prod)

## Getting Your Sumo Logic URL

1. Log in to [Sumo Logic](https://www.sumologic.com/)
2. Go to **Manage Data** > **Collection** > **Collection**
3. Click **Add Source** next to a Hosted Collector
4. Select **HTTP Logs & Metrics**
5. Configure the source:
   - **Name**: LiteLLM Logs
   - **Source Category**: litellm (optional, but helps with queries)
6. Click **Save**
7. Copy the displayed URL - it will look like:
   ```
   https://collectors.sumologic.com/receiver/v1/http/ZaVnC4dhaV39Tn37...
   ```

## Troubleshooting

### Logs not appearing in Sumo Logic

1. **Verify the URL**: Make sure `SUMOLOGIC_WEBHOOK_URL` is set correctly
2. **Check the HTTP Source**: Ensure it's active in Sumo Logic UI
3. **Wait for batching**: Logs are sent in batches, wait 60 seconds
4. **Check for errors**: Enable debug logging in LiteLLM:
   ```python
   litellm.set_verbose = True
   ```

### URL Format

The URL must be the complete HTTP Source URL from Sumo Logic:
- ✅ Correct: `https://collectors.sumologic.com/receiver/v1/http/ZaVnC4dhaV39Tn37...`

### No authentication errors

If you get authentication errors, regenerate the HTTP Source URL in Sumo Logic:
1. Go to your HTTP Source in Sumo Logic
2. Click the settings icon
3. Click **Show URL**
4. Click **Regenerate URL**
5. Update your `SUMOLOGIC_WEBHOOK_URL` environment variable

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
