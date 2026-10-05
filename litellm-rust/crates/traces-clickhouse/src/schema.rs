use std::time::Duration;

use litellm_http::Client;
use litellm_migrate::Migration;
use serde::{Deserialize, Serialize};

use super::{Connection, Error};

const SCHEMA_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const MIGRATIONS: &[Migration] = litellm_migrate::migrate!("migrations");

/// Records the version of every migration applied to a database, so each migration runs once per
/// database and the migration files stay append-only.
const LEDGER: &str = "schema_migrations";

/// Retention is configuration, not history: the TTL of every live table follows `retention_days`
/// on each start. The migrations that first set a TTL remain the record of its introduction.
const RETENTION: [(&str, &str); 3] = [
    ("otel_traces", "toDateTime(Timestamp)"),
    ("spend_logs", "toDateTime(start_time)"),
    ("spans_core", "toDateTime(Timestamp)"),
];

fn render(migration: &Migration, database: &str, retention_days: u32) -> String {
    migration
        .sql
        .replace("{database}", database)
        .replace("{retention_days}", &retention_days.to_string())
}

fn quoted_database(database: &str, retention_days: u32) -> Result<String, Error> {
    if database.is_empty()
        || !database
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || retention_days == 0
    {
        return Err(Error::InvalidSchema);
    }
    Ok(format!("`{database}`"))
}

/// Every statement that brings an empty database to the current schema, in order.
pub fn schema_statements(database: &str, retention_days: u32) -> Result<Vec<String>, Error> {
    let database = quoted_database(database, retention_days)?;
    Ok(
        std::iter::once(format!("CREATE DATABASE IF NOT EXISTS {database}"))
            .chain(
                MIGRATIONS
                    .iter()
                    .map(|migration| render(migration, &database, retention_days)),
            )
            .collect(),
    )
}

#[derive(Deserialize)]
struct Applied {
    version: u64,
}

pub async fn ensure_schema(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
) -> Result<(), Error> {
    ensure_schema_with_timeout(
        client,
        connection,
        database,
        retention_days,
        SCHEMA_REQUEST_TIMEOUT,
    )
    .await
}

async fn execute(
    client: &Client,
    connection: &Connection,
    statement: String,
    request_timeout: Duration,
) -> Result<String, Error> {
    let response = client
        .post(connection.url().clone())
        .timeout(request_timeout)
        .body(statement)
        .send()
        .await
        .map_err(|_| Error::SchemaTransport)?;
    if !response.status().is_success() {
        return Err(Error::SchemaFailed(response.status().as_u16()));
    }
    response.text().await.map_err(|_| Error::SchemaTransport)
}

async fn ensure_schema_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    let database = quoted_database(database, retention_days)?;
    for statement in [
        format!("CREATE DATABASE IF NOT EXISTS {database}"),
        format!(
            "CREATE TABLE IF NOT EXISTS {database}.{LEDGER} \
             (version UInt64, description String, applied_at DateTime DEFAULT now()) \
             ENGINE = MergeTree ORDER BY version"
        ),
    ] {
        execute(client, connection, statement, request_timeout).await?;
    }
    let applied = execute(
        client,
        connection,
        format!("SELECT version FROM {database}.{LEDGER} FORMAT JSONEachRow"),
        request_timeout,
    )
    .await?;
    let applied = applied
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Applied>(line).map_err(|_| Error::InvalidResponse))
        .collect::<Result<Vec<_>, _>>()?;
    for migration in MIGRATIONS {
        if applied.iter().any(|row| row.version == migration.version) {
            continue;
        }
        let sql = render(migration, &database, retention_days);
        execute(client, connection, sql, request_timeout).await?;
        execute(
            client,
            connection,
            format!(
                "INSERT INTO {database}.{LEDGER} (version, description) VALUES ({}, '{}')",
                migration.version, migration.description
            ),
            request_timeout,
        )
        .await?;
    }
    for (table, expression) in RETENTION {
        execute(
            client,
            connection,
            format!(
                "ALTER TABLE {database}.{table} MODIFY TTL {expression} + INTERVAL {retention_days} DAY"
            ),
            request_timeout,
        )
        .await?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NormalizedFieldDefinition {
    pub name: &'static str,
    pub clickhouse_column: &'static str,
    pub clickhouse_type: &'static str,
    pub meaning: &'static str,
}

pub const NORMALIZED_FIELD_DEFINITIONS: [NormalizedFieldDefinition; 15] = [
    NormalizedFieldDefinition {
        name: "observation_type",
        clickhouse_column: "ObservationType",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Operation recorded by the span, including agent, model, tool, retrieval and evaluation steps",
    },
    NormalizedFieldDefinition {
        name: "wrapper_candidate",
        clickhouse_column: "WrapperCandidate",
        clickhouse_type: "Bool",
        meaning: "Span may only wrap the operation it names; the trace graph decides",
    },
    NormalizedFieldDefinition {
        name: "agent_name",
        clickhouse_column: "AgentName",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent associated with this span",
    },
    NormalizedFieldDefinition {
        name: "framework",
        clickhouse_column: "Framework",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent framework or SDK that emitted this span, e.g. claude-agent-sdk",
    },
    NormalizedFieldDefinition {
        name: "agent_metadata",
        clickhouse_column: "AgentMetadata",
        clickhouse_type: "String",
        meaning: "Typed agent metadata as JSON, including thread, subagent, runtime and repository identity",
    },
    NormalizedFieldDefinition {
        name: "litellm_request_id",
        clickhouse_column: "LiteLLMRequestId",
        clickhouse_type: "String",
        meaning: "LiteLLM response ID used to link a span to a spend log",
    },
    NormalizedFieldDefinition {
        name: "call_keys",
        clickhouse_column: "CallKeys",
        clickhouse_type: "Array(String)",
        meaning: "Model requests the span accounts for, as kind:id (litellm_request, provider_response, transport)",
    },
    NormalizedFieldDefinition {
        name: "call_evidence",
        clickhouse_column: "CallEvidence",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Whether CallKeys are all of the span's requests: complete, partial or unknown",
    },
    NormalizedFieldDefinition {
        name: "model",
        clickhouse_column: "Model",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Model used by this span",
    },
    NormalizedFieldDefinition {
        name: "input_tokens",
        clickhouse_column: "InputTokens",
        clickhouse_type: "UInt32",
        meaning: "Input token count",
    },
    NormalizedFieldDefinition {
        name: "output_tokens",
        clickhouse_column: "OutputTokens",
        clickhouse_type: "UInt32",
        meaning: "Output token count",
    },
    NormalizedFieldDefinition {
        name: "input",
        clickhouse_column: "Input",
        clickhouse_type: "String",
        meaning: "Normalized input payload",
    },
    NormalizedFieldDefinition {
        name: "input_preview",
        clickhouse_column: "InputPreview",
        clickhouse_type: "String",
        meaning: "Latest user message of the input, else the input's first characters",
    },
    NormalizedFieldDefinition {
        name: "output",
        clickhouse_column: "Output",
        clickhouse_type: "String",
        meaning: "Normalized output payload",
    },
    NormalizedFieldDefinition {
        name: "tool_call_id",
        clickhouse_column: "ToolCallId",
        clickhouse_type: "String",
        meaning: "Tool call the span executes, shared by instrumentations recording the same call",
    },
];
