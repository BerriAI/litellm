use std::time::Duration;

use litellm_http::Client;
use litellm_migrate::Migration;
use serde::Deserialize;
use serde_with::{DisplayFromStr, PickFirst, serde_as};

use crate::{Connection, Error, READ_LIMITS, valid_identifier};

pub async fn execute_statement(
    client: &Client,
    connection: &Connection,
    sql: &str,
    timeout: Duration,
) -> Result<(), Error> {
    let body = execute_sql(client, connection, sql, timeout).await?;
    if !body.trim().is_empty() {
        return Err(Error::InvalidResponse);
    }
    Ok(())
}

async fn execute_sql(
    client: &Client,
    connection: &Connection,
    sql: &str,
    timeout: Duration,
) -> Result<String, Error> {
    let mut url = connection.url().clone();
    let pairs: Vec<_> = url
        .query_pairs()
        .filter(|(key, _)| {
            !matches!(
                key.as_ref(),
                "query" | "wait_end_of_query" | "send_progress_in_http_headers" | "async_insert"
            )
        })
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.query_pairs_mut()
        .clear()
        .extend_pairs(pairs)
        .append_pair("wait_end_of_query", "1")
        .append_pair("send_progress_in_http_headers", "0")
        .append_pair("async_insert", "0");
    let mut response = client
        .post(url)
        .timeout(timeout)
        .body(sql.to_owned())
        .send()
        .await
        .map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        return Err(Error::SchemaFailed(response.status().as_u16()));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| Error::Transport)? {
        if body.len() + chunk.len() > READ_LIMITS.response_bytes {
            return Err(Error::ResponseTooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| Error::InvalidResponse)
}

#[serde_as]
#[derive(Deserialize)]
struct Applied {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
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
    let response = execute_sql(
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
