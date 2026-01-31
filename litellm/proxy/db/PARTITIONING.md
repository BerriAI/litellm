# PostgreSQL Partitioning Guide for LiteLLM Spend Logs

This guide provides instructions and SQL templates for converting the `LiteLLM_SpendLogs` table to a partitioned table in PostgreSQL. This is recommended for production environments with over 1 million log entries.

## Why Partition?
- **Retention Management**: Dropping a week of logs is a constant-time `DROP TABLE` operation instead of a slow, vacuum-heavy `DELETE`.
- **Performance**: Queries with a time range can skip scanning irrelevant partitions (Partition Pruning).
- **Index Scale**: Indexes are built per-partition, remaining small and efficient.

## Migration Steps

> [!WARNING]
> Perform a full database backup before running these migrations.

### 1. Rename Existing Table
```sql
ALTER TABLE "LiteLLM_SpendLogs" RENAME TO "LiteLLM_SpendLogs_legacy";
```

### 2. Create Partitioned Table
```sql
CREATE TABLE "LiteLLM_SpendLogs" (
    request_id          TEXT NOT NULL,
    call_type           TEXT NOT NULL,
    api_key             TEXT DEFAULT '' NOT NULL,
    spend               DOUBLE PRECISION DEFAULT 0.0 NOT NULL,
    total_tokens        INTEGER DEFAULT 0 NOT NULL,
    prompt_tokens       INTEGER DEFAULT 0 NOT NULL,
    completion_tokens   INTEGER DEFAULT 0 NOT NULL,
    "startTime"         TIMESTAMP WITH TIME ZONE NOT NULL,
    "endTime"           TIMESTAMP WITH TIME ZONE NOT NULL,
    "completionStartTime" TIMESTAMP WITH TIME ZONE,
    model               TEXT DEFAULT '' NOT NULL,
    model_id            TEXT DEFAULT '',
    model_group         TEXT DEFAULT '',
    custom_llm_provider TEXT DEFAULT '',
    api_base            TEXT DEFAULT '',
    "user"              TEXT DEFAULT '',
    metadata            JSONB DEFAULT '{}',
    cache_hit           TEXT DEFAULT '',
    cache_key           TEXT DEFAULT '',
    request_tags        JSONB DEFAULT '[]',
    team_id             TEXT,
    organization_id     TEXT,
    end_user            TEXT,
    requester_ip_address TEXT,
    messages            JSONB DEFAULT '{}',
    response            JSONB DEFAULT '{}',
    session_id          TEXT,
    status              TEXT,
    mcp_namespaced_tool_name TEXT,
    agent_id            TEXT,
    proxy_server_request JSONB DEFAULT '{}',
    PRIMARY KEY (request_id, "startTime")
) PARTITION BY RANGE ("startTime");
```

### 3. Create Initial Partitions
Example for the first month of 2026:
```sql
CREATE TABLE "LiteLLM_SpendLogs_y2026m01" PARTITION OF "LiteLLM_SpendLogs"
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

### 4. Optimize with BRIN Index
```sql
CREATE INDEX "spend_logs_start_time_brin" ON "LiteLLM_SpendLogs" USING BRIN ("startTime");
```

### 5. (Optional) Migrate Legacy Data
```sql
INSERT INTO "LiteLLM_SpendLogs" SELECT * FROM "LiteLLM_SpendLogs_legacy";
```

## Automating Partition Management
Consider using `pg_partman` or a simple cron job to create future partitions and drop expired ones.
