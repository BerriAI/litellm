# PostgreSQL-compatible analytics logging

LiteLLM's existing PostgreSQL spend-log writer uses SQL that Hologres does not fully support. This callback writes logs to a separate PG-compatible analytics database with ordinary batched INSERTs, independently of the proxy's `DATABASE_URL`.

The destination must support the PostgreSQL wire protocol, `SET statement_timeout` and the column types below. Hologres 5.1 was tested. Other PG-compatible engines require their own SQL and schema checks.

## Setup

Install `psycopg[binary]>=3.2,<4` and set `POSTGRES_OLAP_DSN` to a PostgreSQL URI or libpq connection string. TLS defaults to `sslmode=verify-full`. Supply the server CA through `sslrootcert` or `PGSSLROOTCERT`. An explicit DSN `sslmode` overrides `PGSSLMODE`; a local database without TLS needs an explicit `sslmode=disable`.

Create the destination table and grant the writer schema access and INSERT permission. Operators manage engine-specific storage, indexes and retention.

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

Create `analytics_logger.py` beside the proxy configuration.

```python
from litellm.integrations.postgres_olap import PostgresOlapLogger

logger = PostgresOlapLogger()
```

```yaml
litellm_settings:
  callbacks: [analytics_logger.logger]
```

Use `PostgresOlapConfig` to change the schema, table or batch settings. Keep credentials in the environment. SDK users can assign the logger to `litellm.callbacks`, call `await GLOBAL_LOGGING_WORKER.flush()` followed by `await logger.aclose()` before exiting.

## Behavior

Success, failure and completed streaming calls retain typed metrics and a JSON-text payload. Content logging is off by default, including metadata, model parameters, responses and free-text errors. Explicit `turn_off_message_logging=False` enables content unless global or per-request redaction applies.

The default buffer holds 16 MiB, limits events to 1 MiB and writes batches of 100 every second. Failed batches retry twice. Crashes, overflow and shutdown deadlines can lose records, and ambiguous retries can duplicate stable `event_id` values. Deduplicate by that field when analyzing logs. `statistics` distinguishes delivered and dropped events; a successful flush means processing finished, including dropped batches.
