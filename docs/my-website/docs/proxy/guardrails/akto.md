# Akto

## Overview
[Akto](https://www.akto.io/) provides API security guardrails and data ingestion for LLM traffic.

Akto now uses a **two-entry guardrail pattern** in LiteLLM:
- `akto-validate` (`pre_call`) for request validation
- `akto-ingest` (`post_call`) for request/response ingestion

There is no `on_flagged` setting anymore.

Use these as two separate guardrails in `config.yaml`:
- `guardrail_name: "akto-validate"`
- `guardrail_name: "akto-ingest"`

## 1. Get Your Akto Credentials

Set up the Akto Guardrail API Service and grab:
- `AKTO_GUARDRAIL_API_BASE` — your Guardrail API Base URL
- `AKTO_API_KEY` — your API key

## 2. Configure in `config.yaml`

### Block + Ingest (recommended)

Use both entries below. This gives you:
- pre-call block decision
- post-call ingestion for allowed traffic

Keep these as two separate entries (`akto-validate` and `akto-ingest`).

```yaml
guardrails:
  - guardrail_name: "akto-validate"
    litellm_params:
      guardrail: akto
      mode: pre_call
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      default_on: true
      unreachable_fallback: fail_closed   # optional: fail_open | fail_closed (default: fail_closed)
      guardrail_timeout: 5                # optional, default: 5
      akto_account_id: "1000000"         # optional, env fallback: AKTO_ACCOUNT_ID
      akto_vxlan_id: "0"                 # optional, env fallback: AKTO_VXLAN_ID

  - guardrail_name: "akto-ingest"
    litellm_params:
      guardrail: akto
      mode: post_call
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      default_on: true
```

### Monitor-only mode

If you only want logging/ingestion and no blocking, keep only `akto-ingest`.

```yaml
guardrails:
  - guardrail_name: "akto-ingest"
    litellm_params:
      guardrail: akto
      mode: post_call
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      default_on: true
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
    "code": "403"
  }
}
```

## 4. How It Works

**Block + Ingest mode:**
```
Request → LiteLLM → Akto guardrail check
  → Allowed  → forward to LLM → ingest response
  → Blocked  → ingest blocked marker → 403 error
```

**Monitor-only mode:**
```
Request → LiteLLM → forward to LLM → get response
  → Send to Akto (guardrails + ingest) → log only
```

## 5. Event behavior

| Entry | LiteLLM hook | Akto call behavior |
|------|---|---|
| `akto-validate` | `pre_call` | Awaited call with `guardrails=true`, `ingest_data=false` |
| `akto-ingest` | `post_call` | Fire-and-forget call with `guardrails=true`, `ingest_data=true` |

When blocked in `pre_call`, LiteLLM sends one fire-and-forget ingest payload with blocked metadata and returns `403`.

## 6. Parameters

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `akto_base_url` | `AKTO_GUARDRAIL_API_BASE` | *required* | Akto Guardrail API Base URL |
| `akto_api_key` | `AKTO_API_KEY` | *required* | API key (sent as `Authorization` header) |
| `akto_account_id` | `AKTO_ACCOUNT_ID` | `1000000` | Akto account id included in payload |
| `akto_vxlan_id` | `AKTO_VXLAN_ID` | `0` | Akto vxlan id included in payload |
| `unreachable_fallback` | — | `fail_closed` | `fail_open` or `fail_closed` |
| `guardrail_timeout` | — | `5` | Timeout in seconds |
| `default_on` | — | `true` (recommended) | Enables the guardrail entry by default |

## 7. Error Handling

| Scenario | `fail_closed` (default) | `fail_open` |
|----------|------------------------|-------------|
| Akto unreachable | ❌ Blocked (503) | ✅ Passes through |
| Akto returns error | ❌ Blocked (503) | ✅ Passes through |
| Guardrail says no | ❌ Blocked (403) | ❌ Blocked (403) |
