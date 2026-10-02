use litellm_http::Client;
use litellm_migrate::Migration;
use serde::Serialize;
use std::time::Duration;

use super::Connection;
use super::Error;

const SCHEMA_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const MIGRATIONS: &[Migration] = litellm_migrate::migrate!("migrations");

pub fn schema_statements(database: &str, retention_days: u32) -> Result<Vec<String>, Error> {
    if database.is_empty()
        || !database
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || retention_days == 0
    {
        return Err(Error::InvalidSchema);
    }
    let database = format!("`{database}`");
    Ok(
        std::iter::once(format!("CREATE DATABASE IF NOT EXISTS {database}"))
            .chain(MIGRATIONS.iter().map(|migration| {
                migration
                    .sql
                    .replace("{database}", &database)
                    .replace("{retention_days}", &retention_days.to_string())
            }))
            .collect(),
    )
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
    for statement in schema_statements(database, retention_days)? {
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

pub const NORMALIZED_FIELD_DEFINITIONS: [NormalizedFieldDefinition; 9] = [
    NormalizedFieldDefinition {
        name: "observation_type",
        clickhouse_column: "ObservationType",
        clickhouse_type: "LowCardinality(String)",
        meaning: "Agent, LLM, tool, chain, or framework span",
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
        name: "litellm_request_id",
        clickhouse_column: "LiteLLMRequestId",
        clickhouse_type: "String",
        meaning: "LiteLLM response ID used to link a span to a spend log",
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
        name: "output",
        clickhouse_column: "Output",
        clickhouse_type: "String",
        meaning: "Normalized output payload",
    },
];
