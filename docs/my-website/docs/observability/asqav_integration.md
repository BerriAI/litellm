import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Asqav - Signed Audit Trail & AI Governance

[Asqav](https://asqav.com/) provides cryptographically signed audit trails and governance for AI applications. Every LLM call is logged with a tamper-evident signature, giving compliance teams verifiable records of model inputs, outputs, and metadata.

**Key Features:**

- **Signed audit logs** - Every call produces a cryptographically signed record, verifiable offline
- **Governance callbacks** - Hook into LiteLLM's callback system with zero changes to your inference code
- **Compliance-ready exports** - Structured logs compatible with SOC 2, ISO 27001, and EU AI Act requirements
- **Model and cost tracking** - Token usage, latency, and cost captured per request

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

## Pre-Requisites

```bash
pip install "asqav[litellm]"
```

Get your API key from [app.asqav.com](https://app.asqav.com/).

## Quick Start

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import os
import litellm
from asqav.extras.litellm import AsqavGuardrail

os.environ["ASQAV_API_KEY"] = "your-asqav-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Register the Asqav callback
asqav_handler = AsqavGuardrail()
litellm.callbacks = [asqav_handler]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize the EU AI Act."}],
)

print(response)
```

Every call is logged to Asqav with a signed audit record. No other changes to your code are needed.

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

### 1. Install the package on the proxy server

```bash
pip install "asqav[litellm]"
```

### 2. Add Asqav to your `config.yaml`

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: ["asqav.extras.litellm.AsqavGuardrail"]

general_settings:
  master_key: "sk-1234"

environment_variables:
  ASQAV_API_KEY: "your-asqav-api-key"
```

### 3. Start the proxy

```bash
litellm --config config.yaml
```

### 4. Test it

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

Signed audit records appear immediately in your [Asqav dashboard](https://app.asqav.com/).

</TabItem>
</Tabs>

## Environment Variables

| Variable         | Description                                  |
| ---------------- | -------------------------------------------- |
| `ASQAV_API_KEY`  | Your Asqav API key for authentication        |

## What Gets Logged?

The [LiteLLM Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) is captured on each successful LLM API call, including:

- Request messages and model parameters
- Response content and finish reason
- Token usage and cost
- Latency metrics
- A cryptographic signature over the full record

Records are retrievable and verifiable via the Asqav API or dashboard.

## Support

- Docs: [docs.asqav.com](https://docs.asqav.com/)
- Issues: [github.com/asqav/asqav-sdk](https://github.com/asqav/asqav-sdk/issues)
