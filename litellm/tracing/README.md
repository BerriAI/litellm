# Lens ledger export

By default the proxy sends each request record to Lens from an in-memory callback, which drops records when Lens is slow or down. Ledger export is the opt-in alternative: every accepted spend delivery also commits a Lens record to Postgres in the same transaction, and a background exporter delivers it from there. Costs reach Lens even across Lens outages and proxy restarts

## Turning it on

```yaml
general_settings:
  lens_ledger_export: true
  # use_spend_logs_ledger must stay on (it is the default)
```

It also needs a database (`DATABASE_URL`) and the usual Lens connection, `LITELLM_LENS_URL` and `LITELLM_LENS_SERVICE_TOKEN`. A path prefix in `LITELLM_LENS_URL` is kept, so `https://host/lens` posts to `https://host/lens/internal/ledger`

## Which path sends a request

Exactly one path produces each request. While ledger export is on, the callback is not registered

The proxy probes `GET /internal/status` in the background and again every 30 seconds, and never blocks startup on it. If Lens is unreachable or returns an error, ledger export stays on and records wait in Postgres. Only a Lens that answers the probe without advertising ledger version 1 (an older Lens) makes the proxy fall back to the callback. When a later probe shows ledger support, the proxy switches back and drains everything that was committed in the meantime

To roll back, set `lens_ledger_export: false`. The callback becomes the producer again and records already committed keep draining whenever Lens supports the ledger

## What is exported

The amount, request id and event id come from the accepted accounting delivery: the billed spend, the spend log request id and the delivery id. They never come from the callback's cost estimate. Each billed retry has its own delivery, so it is its own Lens event. Correlation ids (response id, provider request id, LiteLLM call id, W3C `traceparent` trace and span, session) come from the same row builder the callback uses

Redaction follows the existing rules:

- `turn_off_message_logging` blanks messages, response and request metadata, and strips echoed prompts from provider error strings
- `disable_spend_logs`, and `disable_error_logs` for failed requests, export the cost only: messages, response, metadata and error string are blank and no body is uploaded
- credentials in request metadata are redacted, and `disable_end_user_cost_tracking` drops the end user

The raw body never lands in the local `LiteLLM_SpendLogs` row or the `SPEND_LOGS_URL` export

## Body size limits

Lens advertises an inline limit in its status response (1 MiB per field by default, 64 KiB when it has a payload store). A body within it travels inline in the record

- **Lens without a payload store:** a larger body is replaced by a marked preview that fits the inline limit. The marker reads `…[Lens preview: N bytes omitted]` and keeps JSON valid
- **Lens with a payload store:** the record carries a short preview and the full body goes to a separate outbox, which uploads it after the cost and independently of it
- **Upload limit:** `LITELLM_LENS_PAYLOAD_MAX_BYTES` caps an uploaded body. It defaults to 16 MiB and counts the JSON-encoded size, quotes and escapes included. Values above 30 MiB are lowered to 30 MiB, because Lens accepts at most 32 MiB per request
- **Above the upload limit:** the body is not uploaded. Lens keeps the marked inline preview and the proxy logs a warning naming the request. Streaming uploads are not supported

Once a body is offloaded, the trace detail view and investigation reads, search and citations show the full body. Ad hoc trace SQL over `spend_logs.messages` and `spend_logs.response` only sees the preview

## Tuning

| Variable | Default | Meaning |
| --- | --- | --- |
| `LITELLM_LENS_EXPORT_IDLE_MS` | 250 | How often an idle cost exporter checks for new records (minimum 50) |
| `LITELLM_LENS_EXPORT_BATCH_SIZE` | 100 | Rows claimed per pass (maximum 1000) |
| `LITELLM_LENS_EXPORT_LEASE_SECONDS` | 120 | How long a claim is held before another pod can retry it |
| `LITELLM_LENS_UPLOAD_BATCH_BYTES` | 33554432 | Body bytes one upload pass may hold in memory (a single larger body is still taken alone) |
| `LITELLM_LENS_PAYLOAD_MAX_BYTES` | 16777216 | Largest body uploaded, JSON-encoded, capped at 30 MiB |

Costs and bodies are exported by separate loops, so a slow payload store never delays costs. Cost requests are split at 8 MiB

## Failures and recovery

A record leaves Postgres only when Lens confirms it stored it. Timeouts, 5xx, 404 and similar errors retry with backoff capped at 5 minutes, without limit, and are never deleted

A row is parked only when retrying cannot help:

| `last_error` | Table | Cause |
| --- | --- | --- |
| `Rejected`, `HTTP 400`, `HTTP 413` | `LiteLLM_LensLedgerExport` | Lens refused the record itself |
| `UndecodableEvent` | `LiteLLM_LensLedgerExport` | The stored record is not valid JSON |
| `PayloadStoreDisabled` | `LiteLLM_LensLedgerPayload` | Lens reported it has no payload store |
| `HTTP 400`, `HTTP 413`, `HTTP 422` | `LiteLLM_LensLedgerPayload` | Lens refused the body, for example a checksum mismatch |

Bodies parked as `PayloadStoreDisabled` are requeued automatically when a probe shows the payload store is back. Parked rows are never deleted automatically. To inspect them:

```sql
SELECT last_error, COUNT(*) FROM "LiteLLM_LensLedgerExport" WHERE parked_at IS NOT NULL GROUP BY last_error;
SELECT last_error, COUNT(*) FROM "LiteLLM_LensLedgerPayload" WHERE parked_at IS NOT NULL GROUP BY last_error;
```

After fixing the cause, for example upgrading Lens, requeue them (drop the `last_error` filter to requeue everything):

```sql
UPDATE "LiteLLM_LensLedgerExport"
SET parked_at = NULL, next_attempt_at = CURRENT_TIMESTAMP, last_error = NULL
WHERE parked_at IS NOT NULL AND last_error = 'Rejected';

UPDATE "LiteLLM_LensLedgerPayload"
SET parked_at = NULL, next_attempt_at = CURRENT_TIMESTAMP, last_error = NULL
WHERE parked_at IS NOT NULL AND last_error = 'PayloadStoreDisabled';
```

To give up on a parked row, delete it. Its cost is still in LiteLLM's own spend tables, so this only affects Lens

Agent traces sent straight to Lens over OTLP do not go through this export, and their limits are set by Lens
