import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Respan - LLM Observability & AI Gateway

[Respan](https://respan.ai/) is an AI observability and gateway platform for tracing, evaluating, and optimizing LLM applications. Respan captures every LLM interaction as a span — containing input, output, model, cost, latency, and metadata — and organizes spans into traces (execution trees), threads (conversations), and scores (evaluation results).

Key features:
- **Trace & Monitor** — Real-time dashboard with requests, tokens, latency, cost, and error rates. Per-user analytics with budget and rate limit controls.
- **Evaluate & Optimize** — Offline and online evaluation with LLM evaluators, code evaluators, and human evaluators.
- **Prompt Management** — Versioned prompt templates with playground testing and deploy-without-code-changes.
- **AI Gateway** — Route 250+ models across OpenAI, Anthropic, Google, Azure, and more with automatic logging, fallbacks, retries, load balancing, and caching.

:::info
We want to learn how we can make the callbacks better! Meet the LiteLLM [founders](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version) or
join our [discord](https://discord.gg/wuPM9dRgDw)
:::

## Pre-Requisites

1. Create an account at [platform.respan.ai](https://platform.respan.ai)
2. Generate an API key from the API keys page
3. Add credits or connect a provider key on the Integrations page

## Quick Start

Respan offers two integration approaches with LiteLLM:

1. **Callback-based** (`respan-exporter-litellm`) — Native LiteLLM callback handler for logging
2. **Auto-instrumented tracing** (`respan-ai` + `openinference-instrumentation-litellm`) — OpenTelemetry-based auto-instrumentation

Each approach supports **Gateway mode** (route requests through Respan), **Logging/Tracing mode** (direct provider calls with async logging to Respan), or both.

## Approach 1: Callback-Based Integration

### Installation

```shell
pip install litellm respan-exporter-litellm
```

### Logging Mode

Register the Respan callback to log all completions automatically. Requests go directly to your LLM provider; only logs are sent to Respan.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
import os
from respan_exporter_litellm import RespanLiteLLMCallback

os.environ["RESPAN_API_KEY"] = "your-respan-api-key"  # from https://platform.respan.ai
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Set Respan as a callback
litellm.callbacks = [RespanLiteLLMCallback()]

# All completions are now logged to Respan
response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Set up your config.yaml:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: custom_callbacks.respan_handler

environment_variables:
  RESPAN_API_KEY: "your-respan-api-key"
```

2. Create the callback file (`custom_callbacks.py`):

```python
from respan_exporter_litellm import RespanLiteLLMCallback

respan_handler = RespanLiteLLMCallback()
```

3. Start the proxy:

```bash
litellm --config config.yaml
```

4. Test it:

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

### Gateway Mode

Route LiteLLM requests through Respan's gateway for full feature access — fallbacks, caching, load balancing, and automatic logging. No separate provider API key is needed if you've added one on the Integrations page.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

response = litellm.completion(
    api_key="your-respan-api-key",
    api_base="https://api.respan.ai/api",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/RESPAN_API_KEY
      api_base: https://api.respan.ai/api
```

```bash
litellm --config config.yaml
```

</TabItem>
</Tabs>

### Pass Respan Parameters

<Tabs>
<TabItem value="logging" label="Logging Mode">

Pass Respan-specific parameters via `metadata.respan_params`:

```python
import litellm
from respan_exporter_litellm import RespanLiteLLMCallback

litellm.callbacks = [RespanLiteLLMCallback()]

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    metadata={
        "respan_params": {
            "workflow_name": "simple_logging",
            "span_name": "single_log",
            "customer_identifier": "user-123",
        }
    },
)
```

</TabItem>
<TabItem value="gateway" label="Gateway Mode">

Pass Respan-specific parameters via `extra_body`:

```python
import litellm

response = litellm.completion(
    api_key="your-respan-api-key",
    api_base="https://api.respan.ai/api",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "customer_identifier": "user-123",
        "metadata": {"session_id": "abc123"},
        "thread_identifier": "conversation_456",
    },
)
```

</TabItem>
</Tabs>

### Async Support

The callback automatically handles async completions:

```python
import litellm
from respan_exporter_litellm import RespanLiteLLMCallback

litellm.callbacks = [RespanLiteLLMCallback()]

response = await litellm.acompletion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke"}],
)
```

## Approach 2: Auto-Instrumented Tracing

Uses OpenTelemetry-based auto-instrumentation for richer trace hierarchies with workflows, tasks, and nested spans.

### Installation

```shell
pip install respan-ai openinference-instrumentation-litellm litellm python-dotenv
```

### Tracing Mode

Calls go directly to providers; Respan auto-instruments them for observability.

```python
import os
from dotenv import load_dotenv
load_dotenv()

import litellm
from respan import Respan
from openinference.instrumentation.litellm import LiteLLMInstrumentor

os.environ["RESPAN_API_KEY"] = "your-respan-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

respan = Respan(instrumentations=[LiteLLMInstrumentor()])

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)
print(response.choices[0].message.content)

respan.flush()
```

### Gateway + Tracing

Combine gateway routing with auto-instrumented tracing:

```python
import os
from dotenv import load_dotenv
load_dotenv()

import litellm
from respan import Respan
from openinference.instrumentation.litellm import LiteLLMInstrumentor

respan = Respan(instrumentations=[LiteLLMInstrumentor()])

response = litellm.completion(
    api_key=os.getenv("RESPAN_API_KEY"),
    api_base="https://api.respan.ai/api",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in three languages."}],
)
print(response.choices[0].message.content)

respan.flush()
```

### Structured Tracing with Decorators

Use `@workflow` and `@task` decorators for rich trace hierarchies:

```python
import litellm
from respan import Respan, workflow, task
from openinference.instrumentation.litellm import LiteLLMInstrumentor

respan = Respan(instrumentations=[LiteLLMInstrumentor()])

@task(name="generate_outline")
def outline(topic: str) -> str:
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Create a brief outline about: {topic}"},
        ],
    )
    return response.choices[0].message.content

@workflow(name="content_pipeline")
def pipeline(topic: str):
    plan = outline(topic)
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Write content from this outline: {plan}"},
        ],
    )
    print(response.choices[0].message.content)

pipeline("Benefits of API gateways")
respan.flush()
```

### Per-Request Attributes

Use `propagate_attributes` to attach Respan-specific attributes to spans within a context:

```python
import litellm
from respan import Respan, workflow, propagate_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor

respan = Respan(
    instrumentations=[LiteLLMInstrumentor()],
    metadata={"service": "chat-api", "version": "1.0.0"},
)

@workflow(name="handle_request")
def handle_request(user_id: str, question: str):
    with propagate_attributes(
        customer_identifier=user_id,
        thread_identifier="conv_001",
        metadata={"plan": "pro"},
    ):
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        print(response.choices[0].message.content)
```

## Respan Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `customer_identifier` | `str` | Identifies the end user for per-user analytics and budget controls |
| `thread_identifier` | `str` | Groups related messages into a conversation thread |
| `metadata` | `dict` | Custom key-value pairs for filtering and search |
| `workflow_name` | `str` | Name for the workflow span (logging mode) |
| `span_name` | `str` | Name for the individual span (logging mode) |
| `disable_log` | `bool` | Set to `true` to disable logging for sensitive data |
| `fallback_models` | `list` | Models to fall back to if primary fails (gateway mode) |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RESPAN_API_KEY` | Respan API key (required) | — |
| `RESPAN_BASE_URL` | Custom API base URL | `https://api.respan.ai/api` |

## What Gets Tracked

Each log in Respan captures:
- **Performance** — Latency, time to first token, duration
- **Cost** — Token usage (input/output/total), cost
- **Identity** — Customer identifier, metadata, thread identifier
- **Content** — Input messages, output response, model
- **Status** — Success/error, error details

## Support & Talk to Founders

- [Schedule Demo](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord](https://discord.gg/wuPM9dRgDw)

For Respan-specific support, visit the [Respan documentation](https://respan.ai/docs).
