# Akto

## Overview
[Akto](https://www.akto.io/) provides API security guardrails and data ingestion for LLM traffic. It validates requests against configurable guardrail policies and optionally ingests request/response pairs for API observability.

Akto supports two operating modes via `on_flagged`:
- **`block`**: Pre-call guardrail validation + post-call data ingestion (blocking)
- **`monitor`**: Single post-call with guardrails + data ingestion (non-blocking, log-only)

## 1. Get Your Akto Credentials

Set up your Akto Data Ingestion Service and obtain:
- `AKTO_GUARDRAIL_API_BASE` — Your Akto Data Ingestion Service endpoint
- `AKTO_API_KEY` — Your API key (**required** for authentication)

## 2. Define Akto Guardrail in `config.yaml`

### Block Mode (Blocking) — Recommended

Pre-call guardrail check blocks violating requests. Post-call ingests data for observability.

```yaml
guardrails:
  - guardrail_name: "akto-guard"
    litellm_params:
      guardrail: akto
      mode: [pre_call, post_call]
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY            # required
      on_flagged: block                                # default: block
      akto_account_id: os.environ/AKTO_ACCOUNT_ID      # optional, default: 1000000
      akto_vxlan_id: os.environ/AKTO_VXLAN_ID           # optional
      unreachable_fallback: fail_open                   # optional, default: fail_closed
      guardrail_timeout: 10                             # optional, default: 5s
```

### Monitor Mode (Non-Blocking)

Single post-call with guardrails + data ingestion. Does not block requests.

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

## 3. Test Request

Send a test request to verify the guardrail is working:

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

### Blocked Request Example

If a request violates Akto guardrail policies (`on_flagged: block`):

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

### Block Mode Flow

```
User Request → LiteLLM Proxy
  → Pre-call: POST /api/http-proxy?akto_connector=litellm&guardrails=true
    → Allowed=true  → forward to LLM → get response
      → Post-call: POST /api/http-proxy?akto_connector=litellm&ingest_data=true
    → Allowed=false → ingest blocked details → return error to user
```

### Monitor Mode Flow

```
User Request → LiteLLM Proxy → forward to LLM → get response
  → Post-call: POST /api/http-proxy?akto_connector=litellm&guardrails=true&ingest_data=true
  → If flagged: logged only, response already sent to user
```

## 5. Configuration Reference

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `akto_base_url` | `AKTO_DATA_INGESTION_URL` | *required* | Akto Data Ingestion Service URL |
| `akto_api_key` | `AKTO_API_KEY` | *required* | API key for authentication (sent as `Authorization: Bearer` header) |
| `on_flagged` | `AKTO_ON_FLAGGED` | `block` | `block`: blocking pre-call + post-call ingest. `monitor`: single post-call log-only |
| `akto_account_id` | `AKTO_ACCOUNT_ID` | `1000000` | Akto account ID |
| `akto_vxlan_id` | `AKTO_VXLAN_ID` | `0` | VXLAN ID for traffic identification |
| `unreachable_fallback` | — | `fail_closed` | `fail_open` or `fail_closed` when Akto is unreachable |
| `guardrail_timeout` | — | `5` | HTTP timeout in seconds for Akto service calls |

## 6. Error Handling

| Scenario | `fail_closed` (default) | `fail_open` |
|----------|------------------------|-------------|
| Akto is unreachable | ❌ Request blocked (503) | ✅ Request passes through |
| Akto returns error | ❌ Request blocked (503) | ✅ Request passes through |
| Guardrail says Allowed=false | ❌ Request blocked (400) + details ingested | ❌ Request blocked (400) + details ingested |
