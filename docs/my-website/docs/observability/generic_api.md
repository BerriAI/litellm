# Generic API Callback (Webhook)

Send LiteLLM logs to any HTTP endpoint.

:::tip Pre-configured Webhook Integrations

Looking for a specific service? Check out our **[Webhook Integrations](./webhook_integrations/index.md)** for pre-configured options like Sumo Logic, Rubrik, and more!

:::

## Quick Start

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: ["custom_api_name"]

callback_settings:
  custom_api_name:
    callback_type: generic_api
    endpoint: https://your-endpoint.com/logs
    headers:
      Authorization: Bearer sk-1234
```

## Configuration

### Basic Setup

```yaml
callback_settings:
  <callback_name>:
    callback_type: generic_api
    endpoint: https://your-endpoint.com  # required
    headers:                              # optional
      Authorization: Bearer <token>
      Custom-Header: value
    event_types:                          # optional, defaults to all events
      - llm_api_success
      - llm_api_failure
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `callback_type` | string | Yes | Must be `generic_api` |
| `endpoint` | string | Yes | HTTP endpoint to send logs to |
| `headers` | dict | No | Custom headers for the request |
| `event_types` | list | No | Filter events: `llm_api_success`, `llm_api_failure`. Defaults to all events. |

## Pre-configured Callbacks

Use built-in configurations from `generic_api_compatible_callbacks.json`. See [Webhook Integrations](./webhook_integrations/index.md) for the full list.

```yaml
litellm_settings:
  callbacks: ["rubrik"]  # loads pre-configured settings

environment_variables:
  RUBRIK_API_KEY: sk-1234
  RUBRIK_WEBHOOK_URL: https://your-rubrik-webhook.com
```

**Available pre-configured callbacks:**
- `rubrik` - Rubrik data security platform
- `sumologic` - Sumo Logic logging platform
- [Add your own!](../contribute_integration/custom_webhook_api)

## Payload Format

Logs are sent as `StandardLoggingPayload` [objects](https://docs.litellm.ai/docs/proxy/logging_spec) in JSON format:

```json
[
  {
    "id": "chatcmpl-123",
    "call_type": "litellm.completion",
    "model": "gpt-3.5-turbo",
    "messages": [...],
    "response": {...},
    "usage": {...},
    "cost": 0.0001,
    "startTime": "2024-01-01T00:00:00",
    "endTime": "2024-01-01T00:00:01",
    "metadata": {...}
  }
]
```

## Environment Variables

Set via environment variables instead of config:

```bash
export GENERIC_LOGGER_ENDPOINT=https://your-endpoint.com
export GENERIC_LOGGER_HEADERS="Authorization=Bearer token,Custom-Header=value"
```

## Batch Settings

Control batching behavior (inherits from `CustomBatchLogger`):

```yaml
callback_settings:
  my_api:
    callback_type: generic_api
    endpoint: https://your-endpoint.com
    batch_size: 100        # default: 100
    flush_interval: 60     # seconds, default: 60
```


