# Guardrails on Pass-Through Endpoints

import Image from '@theme/IdealImage';

## Overview

| Property | Details |
|----------|---------|
| Description | Enable guardrail execution on LiteLLM pass-through endpoints with opt-in activation and automatic inheritance from org/team/key levels |
| Supported Guardrails | All LiteLLM guardrails (Bedrock, Aporia, Lakera, etc.) |
| Default Behavior | Guardrails are **disabled** on pass-through endpoints unless explicitly enabled |

## Quick Start

You can configure guardrails on pass-through endpoints either via the **UI** (recommended) or **config file**.

### Using the UI

#### 1. Navigate to Pass-Through Endpoints

Go to **Models + Endpoints** → Click **+ Add Pass-Through Endpoint**

<Image img={require('../../img/pt_guard1.png')} alt="Add guardrails to pass-through endpoint" />

Scroll to the **Guardrails** section and select which guardrails to enforce.

:::tip Default Behavior
By default, you don't need to specify fields - LiteLLM will JSON dump the entire request/response payload and send it to the guardrail.
:::

#### 2. Target Specific Fields (Optional)

<Image img={require('../../img/pt_guard2.png')} alt="Configure field-level targeting" />

To check only specific fields instead of the entire payload:

1. Select your guardrails
2. In **Field Targeting (Optional)**, specify fields for each guardrail
3. Use the quick-add buttons (`+ query`, `+ documents[*]`) or type custom JSONPath expressions
4. **Request Fields (pre_call)**: Fields to check before sending to target API
5. **Response Fields (post_call)**: Fields to check in the response from target API

**Example**: In the screenshot above, we set `query` as a request field, so only the `query` field is sent to the guardrail instead of the entire request.

---

### Using Config File

#### 1. Define guardrails and pass-through endpoint

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "pii-guard"
    litellm_params:
      guardrail: bedrock
      mode: pre_call
      guardrailIdentifier: "your-guardrail-id"
      guardrailVersion: "1"

general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
      guardrails:
        pii-guard:
```

#### 2. Start proxy

```bash
litellm --config config.yaml
```

#### 3. Test request

```bash
curl -X POST "http://localhost:4000/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France."]
  }'
```

---

## Opt-In Behavior

| Configuration | Behavior |
|--------------|----------|
| `guardrails` not set | No guardrails execute (default) |
| `guardrails` set | All org/team/key + pass-through guardrails execute |

When guardrails are enabled, the system collects and executes:
- Org-level guardrails
- Team-level guardrails  
- Key-level guardrails
- Pass-through specific guardrails

---


## How It Works

The diagram below shows what happens when a client makes a request to `/special/rerank` - a pass-through endpoint configured with guardrails in your `config.yaml`.

When guardrails are configured on a pass-through endpoint:
1. **Pre-call guardrails** run on the request before forwarding to the target API
2. If `request_fields` is specified (e.g., `["query"]`), only those fields are sent to the guardrail. Otherwise, the entire request payload is evaluated.
3. The request is forwarded to the target API only if guardrails pass
4. **Post-call guardrails** run on the response from the target API
5. If `response_fields` is specified (e.g., `["results[*].text"]`), only those fields are evaluated. Otherwise, the entire response is checked.

:::info
If the `guardrails` block is omitted or empty in your pass-through endpoint config, the request skips the guardrail flow entirely and goes directly to the target API.
:::

```mermaid
sequenceDiagram
    participant Client
    box rgb(200, 220, 255) LiteLLM Proxy
        participant PassThrough as Pass-through Endpoint
        participant Guardrails
    end
    participant Target as Target API (Cohere, etc.)

    Client->>PassThrough: POST /special/rerank
    Note over PassThrough,Guardrails: Collect passthrough + org/team/key guardrails
    PassThrough->>Guardrails: Run pre_call (request_fields or full payload)
    Guardrails-->>PassThrough: ✓ Pass / ✗ Block
    PassThrough->>Target: Forward request
    Target-->>PassThrough: Response
    PassThrough->>Guardrails: Run post_call (response_fields or full payload)
    Guardrails-->>PassThrough: ✓ Pass / ✗ Block
    PassThrough-->>Client: Return response (or error)
```

---

## Field-Level Targeting

Target specific JSON fields instead of the entire request/response payload.

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "pii-detection"
    litellm_params:
      guardrail: bedrock
      mode: pre_call
      guardrailIdentifier: "pii-guard-id"
      guardrailVersion: "1"

  - guardrail_name: "content-moderation"
    litellm_params:
      guardrail: bedrock
      mode: post_call
      guardrailIdentifier: "content-guard-id"
      guardrailVersion: "1"

general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
      guardrails:
        pii-detection:
          request_fields: ["query", "documents[*].text"]
        content-moderation:
          response_fields: ["results[*].text"]
```

### Field Options

| Field | Description |
|-------|-------------|
| `request_fields` | JSONPath expressions for input (pre_call) |
| `response_fields` | JSONPath expressions for output (post_call) |
| Neither specified | Guardrail runs on entire payload |

### JSONPath Examples

| Expression | Matches |
|------------|---------|
| `query` | Single field named `query` |
| `documents[*].text` | All `text` fields in `documents` array |
| `messages[*].content` | All `content` fields in `messages` array |

---

## Configuration Examples

### Single guardrail on entire payload

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "pii-detection"
    litellm_params:
      guardrail: bedrock
      mode: pre_call
      guardrailIdentifier: "your-id"
      guardrailVersion: "1"

general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      guardrails:
        pii-detection:
```

### Multiple guardrails with mixed settings

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "pii-detection"
    litellm_params:
      guardrail: bedrock
      mode: pre_call
      guardrailIdentifier: "pii-id"
      guardrailVersion: "1"

  - guardrail_name: "content-moderation"
    litellm_params:
      guardrail: bedrock
      mode: post_call
      guardrailIdentifier: "content-id"
      guardrailVersion: "1"

  - guardrail_name: "prompt-injection"
    litellm_params:
      guardrail: lakera
      mode: pre_call
      api_key: os.environ/LAKERA_API_KEY

general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      guardrails:
        pii-detection:
          request_fields: ["input", "query"]
        content-moderation:
        prompt-injection:
          request_fields: ["messages[*].content"]
```
