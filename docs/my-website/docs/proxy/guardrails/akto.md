# Akto

## Overview
[Akto](https://www.akto.io/) provides API security guardrails and data ingestion for LLM traffic. It validates requests against configurable guardrail policies and optionally ingests request/response pairs for API observability.

Akto supports two operating modes:
- **Sync (blocking)**: Pre-call guardrail validation + post-call data ingestion
- **Async (non-blocking)**: Single post-call with guardrails + data ingestion (log-only)

## 1. Get Your Akto Credentials

Set up your Akto Data Ingestion Service and obtain:
- `AKTO_DATA_INGESTION_URL` — Your Akto Data Ingestion Service endpoint
- `AKTO_API_KEY` — Your API key (optional, depends on your setup)

## 2. Define Akto Guardrail in `config.yaml`

### Sync Mode (Blocking) — Recommended

Pre-call guardrail check blocks violating requests. Post-call ingests data for observability.

```yaml
guardrails:
  - guardrail_name: "akto-guard"
    litellm_params:
      guardrail: akto
      mode: "pre_call"
      api_base: os.environ/AKTO_DATA_INGESTION_URL
      api_key: os.environ/AKTO_API_KEY                # optional
      sync_mode: true                                  # default: true
      akto_account_id: os.environ/AKTO_ACCOUNT_ID      # optional, default: 1000000
      akto_vxlan_id: os.environ/AKTO_VXLAN_ID           # optional
      unreachable_fallback: fail_open                   # optional, default: fail_closed
```

### Async Mode (Non-Blocking)

Single post-call with guardrails + data ingestion. Does not block requests.

```yaml
guardrails:
  - guardrail_name: "akto-guard-async"
    litellm_params:
      guardrail: akto
      mode: "post_call"
      api_base: os.environ/AKTO_DATA_INGESTION_URL
      api_key: os.environ/AKTO_API_KEY
      sync_mode: false
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

If a request violates Akto guardrail policies (sync mode):

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

### Sync Mode Flow

```
User Request → LiteLLM Proxy
  → Pre-call: POST /api/http-proxy?akto_connector=litellm&guardrails=true
    → Allowed=true  → forward to LLM → get response
      → Post-call: POST /api/http-proxy?akto_connector=litellm&ingest_data=true
    → Allowed=false → ingest blocked details → return error to user
```

### Async Mode Flow

```
User Request → LiteLLM Proxy → forward to LLM → get response
  → Post-call: POST /api/http-proxy?akto_connector=litellm&guardrails=true&ingest_data=true
  → If flagged: logged only, response already sent to user
```

## 5. Configuration Reference

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `api_base` | `AKTO_DATA_INGESTION_URL` | *required* | Akto Data Ingestion Service URL |
| `api_key` | `AKTO_API_KEY` | `None` | API key for authentication |
| `sync_mode` | `AKTO_SYNC_MODE` | `true` | `true`: blocking pre-call + post-call ingest. `false`: single post-call |
| `akto_account_id` | `AKTO_ACCOUNT_ID` | `1000000` | Akto account ID |
| `akto_vxlan_id` | `AKTO_VXLAN_ID` | `0` | VXLAN ID for traffic identification |
| `unreachable_fallback` | — | `fail_closed` | `fail_open` or `fail_closed` when Akto is unreachable |

## 6. Error Handling

| Scenario | `fail_closed` (default) | `fail_open` |
|----------|------------------------|-------------|
| Akto is unreachable | ❌ Request blocked | ✅ Request passes through |
| Akto returns error | ❌ Request blocked | ✅ Request passes through |
| Guardrail says Allowed=false | ❌ Request blocked + details ingested | ❌ Request blocked + details ingested |
