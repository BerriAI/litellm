# Akto

## Overview
[Akto](https://www.akto.io/) provides monitoring and guardrails for AI/ML workloads.

The Akto guardrail supports two modes:
- `pre_call` — validates requests and blocks if flagged (sync)
- `logging_only` — non-blocking ingestion of request+response for monitoring (async)

Use them together for full protection, or `logging_only` alone for monitor-only mode.

## 1. Get Your Akto Credentials

Set up the Akto Guardrail API Service and grab:
- `AKTO_GUARDRAIL_API_BASE` — your Guardrail API Base URL
- `AKTO_API_KEY` — your API key

## 2. Configure in `config.yaml`

### Block + Ingest (recommended)

A single guardrail entry with both modes. Requests are validated before the LLM call, and allowed traffic is ingested after the response.

```yaml
guardrails:
  - guardrail_name: "akto-guardrail"
    litellm_params:
      guardrail: akto
      mode: [pre_call, logging_only]
      akto_base_url: os.environ/AKTO_GUARDRAIL_API_BASE
      akto_api_key: os.environ/AKTO_API_KEY
      default_on: true
      unreachable_fallback: fail_closed   # optional: fail_open | fail_closed (default: fail_closed)
      guardrail_timeout: 5                # optional, default: 5
      akto_account_id: "1000000"         # optional, env fallback: AKTO_ACCOUNT_ID
      akto_vxlan_id: "0"                 # optional, env fallback: AKTO_VXLAN_ID
```

### Monitor-only mode

No blocking — just ingest all traffic for monitoring.

```yaml
guardrails:
  - guardrail_name: "akto-monitor"
    litellm_params:
      guardrail: akto
      mode: logging_only
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
    "message": "Blocked by Akto Guardrails",
    "type": "None",
    "param": "None",
    "code": "403"
  }
}
```

## 4. How It Works

**Block + Ingest mode (`pre_call` + `logging_only`):**
```
Request → LiteLLM → Akto guardrail check (pre_call, awaited)
  → Allowed  → LLM call → response → Akto ingest (logging_only, fire-and-forget)
  → Blocked  → Akto ingest blocked marker (fire-and-forget) → 403 error
```

**Monitor-only mode (`logging_only`):**
```
Request → LiteLLM → LLM call → response → Akto ingest (fire-and-forget)
```

## 5. Event behavior

| Mode | LiteLLM hook | Akto call | Blocking |
|------|---|---|---|
| `pre_call` | `apply_guardrail` | Awaited: `guardrails=true`, `ingest_data=false` | Yes |
| `logging_only` | `async_log_success_event` | Fire-and-forget: `guardrails=false`, `ingest_data=true` | No |

- Blocked requests produce one fire-and-forget ingest with `statusCode: 403`.
- Allowed requests produce one fire-and-forget ingest with request + response.
- No duplicate messages — each request produces exactly one ingestion call.

## 6. Parameters

| Parameter | Env Variable | Default | Description |
|-----------|-------------|---------|-------------|
| `akto_base_url` | `AKTO_GUARDRAIL_API_BASE` | *required* | Akto Guardrail API Base URL |
| `akto_api_key` | `AKTO_API_KEY` | *required* | API key (sent as `Authorization` header) |
| `akto_account_id` | `AKTO_ACCOUNT_ID` | `1000000` | Akto account id included in payload |
| `akto_vxlan_id` | `AKTO_VXLAN_ID` | `0` | Akto vxlan id included in payload |
| `unreachable_fallback` | — | `fail_closed` | `fail_open` or `fail_closed` |
| `guardrail_timeout` | — | `5` | Timeout in seconds for pre_call validation |
| `default_on` | — | `true` (recommended) | Enables the guardrail by default |

## 7. Error Handling

| Scenario | `fail_closed` (default) | `fail_open` |
|----------|------------------------|-------------|
| Akto unreachable | Blocked (503) | Passes through |
| Akto returns error | Blocked (503) | Passes through |
| Guardrail says blocked | Blocked (403) | Blocked (403) |
