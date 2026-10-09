use litellm_http::Client;
use litellm_storage_clickhouse::{ClickHouseMigrate, execute_statement, storage_error};
use serde::Serialize;
use sqlx::migrate::Migrator;
use std::time::Duration;

use super::{Connection, Error};

const SCHEMA_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

static MIGRATOR: Migrator = Migrator {
    ignore_missing: true,
    locking: false,
    ..sqlx::migrate!("./migrations")
};

const RETENTION: [(&str, &str); 4] = [
    ("otel_traces", "toDateTime(Timestamp)"),
    ("agent_traces_by_key", "toDateTime(StartTs)"),
    ("spend_logs", "toDateTime(start_time)"),
    ("lens_feedback", "toDateTime(CreatedAt)"),
];

fn validate_schema(database: &str, retention_days: u32) -> Result<(), Error> {
    if database.is_empty()
        || !database
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || retention_days == 0
    {
        return Err(Error::InvalidSchema);
    }
    Ok(())
}

fn render(sql: &str, database: &str, retention_days: u32) -> String {
    sql.replace("{database}", database)
        .replace("{retention_days}", &retention_days.to_string())
}

pub fn schema_statements(database: &str, retention_days: u32) -> Result<Vec<String>, Error> {
    validate_schema(database, retention_days)?;
    let database = format!("`{database}`");
    Ok(
        std::iter::once(format!("CREATE DATABASE IF NOT EXISTS {database}"))
            .chain(
                MIGRATOR
                    .migrations
                    .iter()
                    .map(|migration| render(migration.sql.as_str(), &database, retention_days)),
            )
            .collect(),
    )
}

pub async fn apply_migrations(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
) -> Result<(), Error> {
    apply_migrations_with_timeout(
        client,
        connection,
        database,
        retention_days,
        SCHEMA_REQUEST_TIMEOUT,
    )
    .await
}

async fn apply_migrations_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    validate_schema(database, retention_days)?;
    let quoted_database = format!("`{database}`");
    let mut adapter = ClickHouseMigrate::new(
        client,
        connection,
        database,
        |sql| render(sql, &quoted_database, retention_days),
        request_timeout,
    )?;
    MIGRATOR
        .run_direct(None, &mut adapter, false)
        .await
        .map_err(|error| match storage_error(&error) {
            Some(storage_error) => Error::Storage(storage_error.clone()),
            None => Error::Migration(error),
        })
}

pub async fn reconcile_retention(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
) -> Result<(), Error> {
    reconcile_retention_with_timeout(
        client,
        connection,
        database,
        retention_days,
        SCHEMA_REQUEST_TIMEOUT,
    )
    .await
}

async fn reconcile_retention_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    validate_schema(database, retention_days)?;
    let database = format!("`{database}`");
    for (table, expression) in RETENTION {
        execute_statement(
            client,
            connection,
            &format!(
                "ALTER TABLE {database}.{table} MODIFY TTL {expression} + INTERVAL {retention_days} DAY"
            ),
            request_timeout,
        )
        .await?;
    }
    Ok(())
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

async fn ensure_schema_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    apply_migrations_with_timeout(
        client,
        connection,
        database,
        retention_days,
        request_timeout,
    )
    .await?;
    reconcile_retention_with_timeout(
        client,
        connection,
        database,
        retention_days,
        request_timeout,
    )
    .await
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
