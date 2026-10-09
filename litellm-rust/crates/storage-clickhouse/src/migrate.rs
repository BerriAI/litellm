use std::{
    future::Future,
    pin::Pin,
    time::{Duration, Instant},
};

use litellm_http::Client;
use serde::Deserialize;
use serde_with::{DisplayFromStr, PickFirst, serde_as};
use sqlx::{
    Error as SqlxError,
    migrate::{AppliedMigration, Migrate, MigrateError, Migration},
};

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

/// The startup runner records success after execution, never marks migrations dirty, and skips locks
/// A Keeper-backed or deploy-time runner changes only `apply`, `dirty_version`, and `lock`
pub struct ClickHouseMigrate<'a, R> {
    client: &'a Client,
    connection: &'a Connection,
    database: &'a str,
    render: R,
    timeout: Duration,
}

impl<'a, R> ClickHouseMigrate<'a, R>
where
    R: Fn(&str) -> String + Send + Sync,
{
    pub fn new(
        client: &'a Client,
        connection: &'a Connection,
        database: &'a str,
        render: R,
        timeout: Duration,
    ) -> Result<Self, Error> {
        if !valid_identifier(database) {
            return Err(Error::InvalidSchema);
        }
        Ok(Self {
            client,
            connection,
            database,
            render,
            timeout,
        })
    }
}

#[serde_as]
#[derive(Deserialize)]
struct Applied {
    #[serde_as(as = "PickFirst<(_, DisplayFromStr)>")]
    version: i64,
    checksum: String,
}

fn migrate_error(error: Error) -> MigrateError {
    MigrateError::Execute(SqlxError::AnyDriverError(Box::new(error)))
}

fn migrate_execution_error(error: Error, version: i64) -> MigrateError {
    MigrateError::ExecuteMigration(SqlxError::AnyDriverError(Box::new(error)), version)
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Error> {
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(Error::InvalidResponse);
    }
    pairs
        .iter()
        .map(|pair| {
            let high = decode_hex_digit(pair[0]).ok_or(Error::InvalidResponse)?;
            let low = decode_hex_digit(pair[1]).ok_or(Error::InvalidResponse)?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn decode_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn escape_sql_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

type MigrateFuture<'e, T> = Pin<Box<dyn Future<Output = T> + Send + 'e>>;

impl<R> Migrate for ClickHouseMigrate<'_, R>
where
    R: Fn(&str) -> String + Send + Sync,
{
    fn create_schema_if_not_exists<'e>(
        &'e mut self,
        schema_name: &'e str,
    ) -> MigrateFuture<'e, Result<(), MigrateError>> {
        Box::pin(async move {
            if !valid_identifier(schema_name) {
                return Err(migrate_error(Error::InvalidSchema));
            }
            let statement = format!("CREATE DATABASE IF NOT EXISTS `{schema_name}`");
            execute_statement(self.client, self.connection, &statement, self.timeout)
                .await
                .map_err(migrate_error)
        })
    }

    fn ensure_migrations_table<'e>(
        &'e mut self,
        table_name: &'e str,
    ) -> MigrateFuture<'e, Result<(), MigrateError>> {
        Box::pin(async move {
            let database = format!("`{}`", self.database);
            execute_statement(
                self.client,
                self.connection,
                &format!("CREATE DATABASE IF NOT EXISTS {database}"),
                self.timeout,
            )
            .await
            .map_err(migrate_error)?;
            execute_statement(
                self.client,
                self.connection,
                &format!(
                    "CREATE TABLE IF NOT EXISTS {database}.{table_name} \
                     (version Int64, description String, installed_on DateTime64(3) DEFAULT now64(3), \
                     success Bool, checksum String, execution_time Int64) ENGINE = MergeTree ORDER BY version"
                ),
                self.timeout,
            )
            .await
            .map_err(migrate_error)
        })
    }

    fn dirty_version<'e>(
        &'e mut self,
        _table_name: &'e str,
    ) -> MigrateFuture<'e, Result<Option<i64>, MigrateError>> {
        Box::pin(async { Ok(None) })
    }

    fn list_applied_migrations<'e>(
        &'e mut self,
        table_name: &'e str,
    ) -> MigrateFuture<'e, Result<Vec<AppliedMigration>, MigrateError>> {
        Box::pin(async move {
            let database = format!("`{}`", self.database);
            let statement = format!(
                "SELECT DISTINCT version, checksum FROM {database}.{table_name} \
                 WHERE success ORDER BY version FORMAT JSONEachRow"
            );
            let response = execute_sql(self.client, self.connection, &statement, self.timeout)
                .await
                .map_err(migrate_error)?;
            response
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let row = serde_json::from_str::<Applied>(line)
                        .map_err(|_| migrate_error(Error::InvalidResponse))?;
                    let checksum = decode_hex(&row.checksum).map_err(migrate_error)?;
                    Ok(AppliedMigration {
                        version: row.version,
                        checksum: checksum.into(),
                    })
                })
                .collect()
        })
    }

    fn lock(&mut self) -> MigrateFuture<'_, Result<(), MigrateError>> {
        Box::pin(async { Ok(()) })
    }

    fn unlock(&mut self) -> MigrateFuture<'_, Result<(), MigrateError>> {
        Box::pin(async { Ok(()) })
    }

    fn apply<'e>(
        &'e mut self,
        table_name: &'e str,
        migration: &'e Migration,
    ) -> MigrateFuture<'e, Result<Duration, MigrateError>> {
        Box::pin(async move {
            let started_at = Instant::now();
            let statement = (self.render)(migration.sql.as_str());
            execute_statement(self.client, self.connection, &statement, self.timeout)
                .await
                .map_err(|error| migrate_execution_error(error, migration.version))?;
            let elapsed = started_at.elapsed();
            let execution_time = elapsed.as_nanos().min(i64::MAX as u128) as i64;
            let description = escape_sql_string(&migration.description);
            let checksum = encode_hex(&migration.checksum);
            let database = format!("`{}`", self.database);
            execute_statement(
                self.client,
                self.connection,
                &format!(
                    "INSERT INTO {database}.{table_name} \
                     (version, description, success, checksum, execution_time) \
                     VALUES ({}, '{}', true, '{}', {execution_time})",
                    migration.version, description, checksum
                ),
                self.timeout,
            )
            .await
            .map_err(migrate_error)?;
            Ok(elapsed)
        })
    }

    fn revert<'e>(
        &'e mut self,
        _table_name: &'e str,
        _migration: &'e Migration,
    ) -> MigrateFuture<'e, Result<Duration, MigrateError>> {
        Box::pin(async {
            Err(MigrateError::Execute(SqlxError::AnyDriverError(Box::new(
                std::io::Error::other("ClickHouse migrations are forward-only"),
            ))))
        })
    }
}

pub fn storage_error(error: &MigrateError) -> Option<&Error> {
    let error = match error {
        MigrateError::Execute(error) | MigrateError::ExecuteMigration(error, _) => error,
        _ => return None,
    };
    match error {
        SqlxError::AnyDriverError(error) => error.downcast_ref(),
        _ => None,
    }
}
