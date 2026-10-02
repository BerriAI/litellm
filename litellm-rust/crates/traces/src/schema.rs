use litellm_http::Client;
use litellm_migrate::Migration;
use std::time::Duration;

use crate::Connection;
use crate::Error;

const SCHEMA_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const MIGRATIONS: &[Migration] = litellm_migrate::migrate!("migrations");

pub fn schema_statements(
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
) -> Result<Vec<String>, Error> {
    if database.is_empty()
        || !database
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        || trace_retention_days == 0
        || spend_log_retention_days == 0
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
                    .replace("{trace_retention_days}", &trace_retention_days.to_string())
                    .replace(
                        "{spend_log_retention_days}",
                        &spend_log_retention_days.to_string(),
                    )
            }))
            .collect(),
    )
}

pub async fn ensure_schema(
    client: &Client,
    connection: &Connection,
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
) -> Result<(), Error> {
    ensure_schema_with_timeout(
        client,
        connection,
        database,
        trace_retention_days,
        spend_log_retention_days,
        SCHEMA_REQUEST_TIMEOUT,
    )
    .await
}

async fn ensure_schema_with_timeout(
    client: &Client,
    connection: &Connection,
    database: &str,
    trace_retention_days: u32,
    spend_log_retention_days: u32,
    request_timeout: Duration,
) -> Result<(), Error> {
    for statement in schema_statements(database, trace_retention_days, spend_log_retention_days)? {
        let response = client
            .post(connection.url().clone())
            .timeout(request_timeout)
            .body(statement)
            .send()
            .await
            .map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return Err(Error::SchemaFailed(response.status().as_u16()));
        }
    }
    Ok(())
}
