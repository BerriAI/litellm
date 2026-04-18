import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Asqav - Signed Audit Trail & AI Governance

[Asqav](https://asqav.com/) provides cryptographically signed audit trails for AI applications. Every LLM call is signed server-side with ML-DSA, giving compliance teams verifiable, tamper-evident records of model inputs, outputs, and metadata.

This integration is a self-contained custom callback that uses only `httpx` (already a LiteLLM dependency). No extra packages to install.

**Key Features:**

- Signed audit logs for every call, verifiable from the Asqav dashboard
- Pure-stdlib + httpx: no extra dependencies
- Fail-open: any Asqav outage is logged but never breaks an LLM request
- Captures model, message count, token usage, and latency

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

## Pre-Requisites

1. Create an account at [asqav.com](https://asqav.com/) and copy your API key.
2. Create an agent in the Asqav dashboard and copy its `agent_id`.

```bash
export ASQAV_API_KEY="sk_live_..."
export ASQAV_AGENT_ID="agent_..."
```

## Quick Start

The integration is a single LiteLLM `CustomLogger` subclass. Drop it into your project and register it as a callback.

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
import os
import uuid
import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger


def _latency_ms(start, end):
    """LiteLLM passes datetime objects; older paths may pass floats."""
    if not start or not end:
        return None
    delta = end - start
    if hasattr(delta, "total_seconds"):
        return int(delta.total_seconds() * 1000)
    return int(delta * 1000)


class AsqavLogger(CustomLogger):
    """Posts a signed audit record to Asqav after every LLM call."""

    def __init__(self) -> None:
        self.api_key = os.environ["ASQAV_API_KEY"]
        self.agent_id = os.environ["ASQAV_AGENT_ID"]
        self.base_url = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/api/v1")
        self.session_id = str(uuid.uuid4())
        self._client = httpx.Client(timeout=5.0)

    def _sign(self, action_type: str, context: dict) -> None:
        try:
            self._client.post(
                f"{self.base_url}/agents/{self.agent_id}/sign",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={"action_type": action_type, "context": context, "session_id": self.session_id},
            )
        except Exception as e:
            # Fail-open: never break the LLM request on a signing error.
            print(f"asqav signing failed: {e}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = getattr(response_obj, "usage", None)
        self._sign("llm:call", {
            "model": kwargs.get("model"),
            "message_count": len(kwargs.get("messages") or []),
            "latency_ms": _latency_ms(start_time, end_time),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        })

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._sign("llm:error", {
            "model": kwargs.get("model"),
            "error": str(kwargs.get("exception") or response_obj),
        })


litellm.callbacks = [AsqavLogger()]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize the EU AI Act."}],
)
print(response)
```

Every successful and failed call is now signed in the Asqav dashboard.

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

### 1. Save the callback as a Python file

Create `asqav_callback.py` next to your `config.yaml`:

```python
import os
import uuid
import httpx
from litellm.integrations.custom_logger import CustomLogger


def _latency_ms(start, end):
    if not start or not end:
        return None
    delta = end - start
    if hasattr(delta, "total_seconds"):
        return int(delta.total_seconds() * 1000)
    return int(delta * 1000)


class AsqavLogger(CustomLogger):
    def __init__(self) -> None:
        self.api_key = os.environ["ASQAV_API_KEY"]
        self.agent_id = os.environ["ASQAV_AGENT_ID"]
        self.base_url = os.environ.get("ASQAV_API_URL", "https://api.asqav.com/api/v1")
        self.session_id = str(uuid.uuid4())
        self._client = httpx.Client(timeout=5.0)

    def _sign(self, action_type, context):
        try:
            self._client.post(
                f"{self.base_url}/agents/{self.agent_id}/sign",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={"action_type": action_type, "context": context, "session_id": self.session_id},
            )
        except Exception as e:
            print(f"asqav signing failed: {e}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = getattr(response_obj, "usage", None)
        self._sign("llm:call", {
            "model": kwargs.get("model"),
            "message_count": len(kwargs.get("messages") or []),
            "latency_ms": _latency_ms(start_time, end_time),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        })

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._sign("llm:error", {
            "model": kwargs.get("model"),
            "error": str(kwargs.get("exception") or response_obj),
        })


asqav_logger = AsqavLogger()
```

### 2. Reference it from `config.yaml`

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: asqav_callback.asqav_logger

general_settings:
  master_key: "sk-1234"
```

### 3. Set environment variables and start the proxy

```bash
export ASQAV_API_KEY="sk_live_..."
export ASQAV_AGENT_ID="agent_..."
litellm --config config.yaml
```

### 4. Test it

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

Signed audit records appear immediately in your [Asqav dashboard](https://asqav.com/).

</TabItem>
</Tabs>

## Environment Variables

| Variable          | Required | Description                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| `ASQAV_API_KEY`   | yes      | Your Asqav API key                                                |
| `ASQAV_AGENT_ID`  | yes      | The agent identifier created in your Asqav dashboard              |
| `ASQAV_API_URL`   | no       | Override the API base (defaults to `https://api.asqav.com/api/v1`) |

## What Gets Logged?

Each successful call sends:

- `model`: the LiteLLM model string
- `message_count`: number of messages in the request
- `latency_ms`: round-trip latency
- `prompt_tokens` / `completion_tokens`: usage if reported by the provider

Each failed call sends the model and the error string under `llm:error`.

The Asqav backend signs the record server-side with ML-DSA and returns a verification URL.

## Support

- Docs: [asqav.com](https://asqav.com/)
- Issues: [github.com/jagmarques/asqav-sdk](https://github.com/jagmarques/asqav-sdk/issues)
