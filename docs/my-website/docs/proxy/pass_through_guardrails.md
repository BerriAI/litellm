# Guardrails on Pass-Through Endpoints

Enable guardrail execution on LiteLLM pass-through endpoints.

**Key Behavior**: Pass-through endpoints are **opt-in only** for guardrails. Guardrails configured at org/team/key levels will NOT execute unless explicitly enabled.

## Quick Start

### 1. Define guardrails and pass-through endpoint

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

### 2. Start proxy

```shell
litellm --config config.yaml
```

### 3. Test request

```shell
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
| `guardrails` set with keys | All org/team/key + passthrough guardrails execute |

When guardrails are enabled on a pass-through endpoint, the system collects and executes:
- Org-level guardrails
- Team-level guardrails  
- Key-level guardrails
- Pass-through specific guardrails

---

## Field-Level Targeting

Target specific JSON fields instead of the entire payload:

```yaml showLineNumbers title="config.yaml"
general_settings:
  pass_through_endpoints:
    - path: "/v1/rerank"
      target: "https://api.cohere.com/v1/rerank"
      headers:
        Authorization: "bearer os.environ/COHERE_API_KEY"
      guardrails:
        pii-detection:
          request_fields: ["query", "documents[*].text"]
          response_fields: ["results[*].text"]
        content-moderation:  # runs on entire payload
```

### Field Options

| Field | Description |
|-------|-------------|
| `request_fields` | JSONPath expressions for input (pre_call) |
| `response_fields` | JSONPath expressions for output (post_call) |
| Neither specified | Guardrail runs on entire payload |

### JSONPath Examples

| Expression | Description |
|------------|-------------|
| `query` | Single field |
| `documents[*].text` | All `text` fields in `documents` array |
| `messages[*].content` | All `content` fields in `messages` array |

---

## Configuration Examples

### Single guardrail, entire payload

```yaml showLineNumbers title="config.yaml"
guardrails:
  pii-detection:
```

### Request-only targeting

```yaml showLineNumbers title="config.yaml"
guardrails:
  pii-detection:
    request_fields: ["messages[*].content", "input"]
```

### Response-only targeting

```yaml showLineNumbers title="config.yaml"
guardrails:
  content-moderation:
    response_fields: ["results[*].text", "output"]
```

### Multiple guardrails with mixed settings

```yaml showLineNumbers title="config.yaml"
guardrails:
  pii-detection:
    request_fields: ["input", "query"]
  content-moderation:
  prompt-injection:
    request_fields: ["messages[*].content"]
```

