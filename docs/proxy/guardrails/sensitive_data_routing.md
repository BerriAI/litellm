import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sensitive Data Routing (Built-in Guardrail)

**Built-in guardrail** that detects sensitive data in a request and reroutes it to an on-premise model instead of blocking or redacting it. No external dependencies required.

**When to use?** When sensitive prompts must be served by an on-premise model rather than a cloud provider, and the user workflow has to stay uninterrupted.

## Overview

| Property | Details |
|----------|---------|
| Description | Detects sensitive data with regex / keyword matching and reroutes the request to an on-premise model. Once sensitive data appears in a session, every following turn in that session is also routed on-premise. |
| Guardrail Name | `sensitive_data_routing` |
| Detection Methods | Prebuilt regex patterns, custom regex, keyword matching |
| Action | Reroute to an on-premise model (never blocks or redacts) |
| Supported Modes | `pre_call` |
| Performance | Fast; runs locally, no external API calls |

## How it works

The guardrail runs before model selection. On every request it scans the messages for sensitive data using the patterns and keywords you configure. When a match is found it rewrites the target model to your `on_premise_model` so the request is served on-premise. The prompt is sent through unchanged, so nothing is blocked or redacted and the conversation stays seamless.

With `sticky_session` enabled (the default), the first time sensitive data is seen in a session the session is pinned to the on-premise model. Every later turn in that session is then routed on-premise as well, even turns that contain no sensitive data, so a conversation that once touched sensitive data never leaves the on-premise model. Pinning relies on a stable session id sent by the client (see [Session stickiness](#session-stickiness)).

`on_premise_model` is just a model group in your `model_list`. Point it at whatever on-premise deployment you run (vLLM, Ollama, a self-hosted OpenAI-compatible endpoint, and so on).

## Quick Start

### Step 1: Define the guardrail and an on-premise model in config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: cloud-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: on-prem-model
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-8B-Instruct
      api_base: http://your-on-prem-host:8000/v1

guardrails:
  - guardrail_name: "sensitive-data-routing"
    litellm_params:
      guardrail: sensitive_data_routing
      mode: "pre_call"
      default_on: true

      # The model group (from model_list above) to route sensitive requests to
      on_premise_model: "on-prem-model"

      # Built-in detectors
      prebuilt_patterns:
        - us_ssn
        - credit_card
        - email
      regex_patterns:
        - "project\\s+titan"
      keywords:
        - confidential
        - internal only

      # Keep the whole session on-premise once sensitive data is seen
      sticky_session: true
      session_ttl_seconds: 14400
```

### Step 2: Start the proxy

```bash
litellm --config config.yaml --detailed_debug
```

### Step 3: Send a clean request (served by the cloud model)

```bash showLineNumbers
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cloud-model",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "metadata": {"session_id": "abc-123"}
  }'
```

The response `model` field reflects the cloud model.

### Step 4: Send a request with sensitive data (rerouted on-premise)

```bash showLineNumbers
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cloud-model",
    "messages": [{"role": "user", "content": "My SSN is 123-45-6789, summarize my record"}],
    "metadata": {"session_id": "abc-123"}
  }'
```

The request is served by `on-prem-model`. Because `sticky_session` is on and the same `session_id` is used, every later request on `abc-123` is also served on-premise, even if it contains no sensitive data.

## Configuration

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `on_premise_model` | string | required | Model group (from `model_list`) to route sensitive requests to |
| `prebuilt_patterns` | list[string] | none | Built-in pattern names to match (for example `us_ssn`, `credit_card`, `email`). Same library as the [LiteLLM Content Filter](./litellm_content_filter) |
| `regex_patterns` | list[string] | none | Custom regular expressions; a match in any message reroutes the request |
| `keywords` | list[string] | none | Case-insensitive keywords; a match in any message reroutes the request |
| `sticky_session` | bool | `true` | Keep the whole session on-premise after sensitive data is first detected |
| `session_ttl_seconds` | int | `14400` | How long a session stays pinned on-premise after detection |

At least one of `prebuilt_patterns`, `regex_patterns`, or `keywords` is required.

## Session stickiness

Stickiness pins a session to the on-premise model after the first detection. The session is identified by `litellm_session_id`, `metadata.session_id`, or `litellm_metadata.session_id` on the request, so the client must send a stable id across turns for stickiness to apply.

When a Redis cache is configured on the proxy, the pin is shared across all proxy workers and instances, so stickiness holds for the whole deployment and not just a single worker.

If no session id is sent, each turn is still evaluated independently, so any turn that itself contains sensitive data is routed on-premise; turns without a session id are not pinned across the conversation.
