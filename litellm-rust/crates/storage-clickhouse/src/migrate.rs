use std::time::Duration;

use litellm_http::Client;
use litellm_migrate::Migration;
use serde::Deserialize;

use crate::{Connection, Error, valid_identifier};

pub async fn execute_statement(
    client: &Client,
    connection: &Connection,
    sql: &str,
    timeout: Duration,
) -> Result<String, Error> {
    let response = client
        .post(connection.url().clone())
        .timeout(timeout)
        .body(sql.to_owned())
        .send()
        .await
        .map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        return Err(Error::SchemaFailed(response.status().as_u16()));
    }
    response.text().await.map_err(|_| Error::Transport)
}

#[derive(Deserialize)]
struct Applied {
    version: u64,
    checksum: String,
}

pub async fn apply_migrations(
    client: &Client,
    connection: &Connection,
    database: &str,
    migrations: &[Migration],
    render: impl Fn(&Migration) -> String,
    timeout: Duration,
) -> Result<(), Error> {
    if !valid_identifier(database) {
        return Err(Error::InvalidSchema);
    }
    let database = format!("`{database}`");
    for statement in [
        format!("CREATE DATABASE IF NOT EXISTS {database}"),
        format!(
            "CREATE TABLE IF NOT EXISTS {database}.schema_migrations \
             (version UInt64, description String, checksum String, applied_at DateTime DEFAULT now()) \
             ENGINE = MergeTree ORDER BY version"
        ),
    ] {
        execute_statement(client, connection, &statement, timeout).await?;
    }
    let response = execute_statement(
        client,
        connection,
        &format!("SELECT version, checksum FROM {database}.schema_migrations FORMAT JSONEachRow"),
        timeout,
    )
    .await?;
    let applied = response
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Applied>(line).map_err(|_| Error::InvalidResponse))
        .collect::<Result<Vec<_>, _>>()?;
    let pending = litellm_migrate::pending(
        migrations,
        applied
            .iter()
            .map(|row| (row.version, row.checksum.as_str())),
    )?;
    for migration in pending {
        execute_statement(client, connection, &render(migration), timeout).await?;
        execute_statement(
            client,
            connection,
            &format!(
                "INSERT INTO {database}.schema_migrations (version, description, checksum) \
                 VALUES ({}, '{}', '{}')",
                migration.version, migration.description, migration.checksum
            ),
            timeout,
        )
        .await?;
    }
    Ok(())
}
