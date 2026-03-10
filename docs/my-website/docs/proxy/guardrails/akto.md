# Akto

## Overview
[Akto](https://www.akto.io/) provides API security guardrails and data ingestion for LLM traffic.

Two modes via `on_flagged`:
- **`block`** — Pre-call guardrail check + post-call data ingestion (blocks violating requests)
- **`monitor`** — Post-call only, logs violations without blocking

## 1. Get Your Akto Credentials

Set up the Akto Guardrail API Service and grab:
- `AKTO_GUARDRAIL_API_BASE` — your Guardrail API Base URL
- `AKTO_API_KEY` — your API key

## 2. Configure in `config.yaml`

### Block Mode (recommended)

Checks requests before they reach the LLM. Blocked requests get a `400` error.

```yaml
guardrails:
  - guardrail_name: "akto-guard"
    litellm_params:
      guardrail: akto
      mode: [pre_call, post_call]
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      on_flagged: block
      unreachable_fallback: fail_open                   # optional, default: fail_closed
      guardrail_timeout: 10                             # optional, default: 5s
```

### Monitor Mode

Logs everything after the LLM responds. Never blocks requests.

```yaml
guardrails:
  - guardrail_name: "akto-guard-async"
    litellm_params:
      guardrail: akto
      mode: post_call
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      on_flagged: monitor
```

## 3. Test It

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your litellm key>" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

If a request gets blocked:

```json
{
  "error": {
    "message": "Prompt injection detected",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

## 4. How It Works

**Block mode:**
```
Request → LiteLLM → Akto guardrail check
  → Allowed  → forward to LLM → ingest response
  → Blocked  → ingest blocked details → 400 error
```

**Monitor mode:**
```
Request → LiteLLM → forward to LLM → get response
  → Send to Akto (guardrails + ingest) → log only
```

## 5. Parameters

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `akto_base_url` | `AKTO_GUARDRAIL_API_BASE` | *required* | Akto Guardrail API Base URL |
| `akto_api_key` | `AKTO_API_KEY` | *required* | API key (sent as `Authorization` header) |
| `on_flagged` | `AKTO_ON_FLAGGED` | `block` | `block` or `monitor` |
| `unreachable_fallback` | — | `fail_closed` | `fail_open` or `fail_closed` |
| `guardrail_timeout` | — | `5` | Timeout in seconds |

## 6. Error Handling

| Scenario | `fail_closed` (default) | `fail_open` |
|----------|------------------------|-------------|
| Akto unreachable | ❌ Blocked (503) | ✅ Passes through |
| Akto returns error | ❌ Blocked (503) | ✅ Passes through |
| Guardrail says no | ❌ Blocked (400) | ❌ Blocked (400) |
