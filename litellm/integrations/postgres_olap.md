# PostgreSQL protocol log export

Send success and failure logs to an independent PostgreSQL-compatible analytics database, including Hologres. This exporter uses one ordinary multi-row INSERT per batch through Psycopg's client-side parameter binding. It does not require prepared statements, COPY, JSONB types, conflict clauses, foreign keys or LiteLLM database migrations. The destination must support INSERT, SET statement_timeout and the column types below.

Install `psycopg[binary]>=3.2,<4` alongside LiteLLM. Configure a dedicated destination with `POSTGRES_OLAP_DSN`. It accepts a PostgreSQL URI or libpq keyword connection string, including `sslmode=verify-full` and `sslrootcert`. Use the endpoint, CA and authentication required by your database. This setting is independent of the proxy's `DATABASE_URL`.

Create the table once with an administrator. The runtime account only needs schema access and INSERT permission. Schema ownership, partitioning, indexes and retention belong to the operator.

```sql
CREATE TABLE public.litellm_logs (
    event_id TEXT NOT NULL,
    request_id TEXT,
    status TEXT,
    model TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    response_cost DOUBLE PRECISION,
    payload TEXT
);
```

For Hologres, choose column storage and tune distribution and retention for your workload. No primary key is required. Other PG protocol databases need their own schema and SQL compatibility checks.

## SDK

```python
import asyncio
import litellm
from litellm.integrations.postgres_olap import PostgresOlapLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

async def main():
    logger = PostgresOlapLogger()
    litellm.callbacks = [logger]
    try:
        response = await litellm.acompletion(
            model="openai/your-model", messages=[{"role": "user", "content": "Hello"}]
        )
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await logger.aclose()

asyncio.run(main())
```

Drain LiteLLM callbacks before closing the destination. `flush()` waits for events already admitted to this exporter, not callbacks still waiting in LiteLLM. Inspect `statistics` to distinguish delivered and dropped records. A successful flush means all admitted events were processed, including exhausted failures.

## Proxy

Create `analytics_logger.py` beside your proxy configuration.

```python
from litellm.integrations.postgres_olap import PostgresOlapLogger

logger = PostgresOlapLogger()
```

```yaml
litellm_settings:
  callbacks: [analytics_logger.logger]
```

To configure the destination table or batching, construct the logger with `PostgresOlapConfig(dsn=SecretStr(os.environ["POSTGRES_OLAP_DSN"]), table_name="my_logs", batch_size=100)`. Schema and table names are quoted identifiers. Do not put credentials in YAML or source control.

## Analysis

```sql
SELECT model, status, COUNT(*) AS requests,
       SUM(total_tokens) AS tokens, SUM(response_cost) AS cost
FROM public.litellm_logs
GROUP BY model, status;
```

`payload` contains the standard logging object as JSON text, preserving fields beyond the typed analysis columns. Cast it using your engine's JSON functions when needed. Missing metrics remain NULL. Timestamps without timezone information from LiteLLM are interpreted as UTC.

## Delivery and privacy

The exporter handles synchronous and asynchronous success and failure callbacks. Streaming calls use LiteLLM's completed standard logging object. Request and response content is hidden by default. Set `turn_off_message_logging=False` explicitly to store it. Global and per-request message redaction and excluded standard payload fields still apply.

A daemon worker writes batches of at most 100 events every second by default. The queue, including its in-flight batch, is limited to 16 MiB of estimated serialized data. Individual events over 1 MiB and newly arriving events over capacity are dropped. Python object overhead is additional. Serialization runs outside the async event loop. Connections are opened per batch, with a five-second connect timeout and ten-second statement timeout.

Failed batches retry twice and then drop. Error messages report only exception classes, avoiding SQL, payloads and connection strings. An invalid row can reject its entire batch. Retries preserve `event_id`, but an ambiguous commit can produce duplicate rows. Deduplicate by `event_id` in analysis or use a destination-specific ingestion strategy. Request IDs are not event IDs, since multiple attempts can share a request ID.

This is best-effort in-memory export. Process crashes, capacity limits and shutdown deadlines can lose events. `close()` stops admission and waits up to ten seconds for draining. A blocked driver can outlive that deadline in its daemon thread. Use a durable message broker when replay or durable buffering is required.
