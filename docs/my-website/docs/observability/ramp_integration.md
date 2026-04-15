import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Ramp

Send AI usage and cost data to Ramp for automated spend tracking.

[Ramp](https://ramp.com/) is a finance automation platform that helps businesses manage expenses, corporate cards, and vendor payments. With the Ramp callback integration, your LiteLLM AI usage — including token counts, model costs, and request metadata — is automatically sent to Ramp for real-time spend visibility.

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

1. Log in to [Ramp](https://app.ramp.com/) and search for **"LiteLLM"** using the search bar. Click the **LiteLLM** integration result.

> **Note:** Only business owners and admins can access and configure integrations.

2. On the LiteLLM integration page, click the **Connect** button in the top right.

3. In the Connect LiteLLM drawer, click **Generate API Key** to create an API key.

> **Important:** Copy the API key immediately — it won't be shown again. If you lose it, you can revoke the existing key and generate a new one from the integration settings.

```shell
pip install litellm
```

## Quick Start

Set your `RAMP_API_KEY` and add `"ramp"` to your callbacks to start logging LLM usage to Ramp.

<Tabs>
<TabItem value="python" label="SDK">

```python
litellm.callbacks = ["ramp"]
```

```python
import litellm
import os

# Ramp API Key
os.environ["RAMP_API_KEY"] = "your-ramp-api-key"

# LLM API Keys
os.environ['OPENAI_API_KEY'] = ""

# Set ramp as a callback
litellm.callbacks = ["ramp"]

# OpenAI call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi - I'm testing Ramp integration"}
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
  callbacks: ["ramp"]

environment_variables:
  RAMP_API_KEY: os.environ/RAMP_API_KEY
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

LiteLLM sends the [Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) to Ramp on successful LLM API calls, which includes:

- **Request details**: Model, messages, parameters
- **Response details**: Completion text, token usage, latency
- **Metadata**: User ID, custom metadata, timestamps
- **Cost tracking**: Response cost based on token usage

## Authentication

Set the `RAMP_API_KEY` environment variable with your Ramp API key.

| Environment Variable | Description |
|---|---|
| `RAMP_API_KEY` | Your Ramp API key (required) |

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
