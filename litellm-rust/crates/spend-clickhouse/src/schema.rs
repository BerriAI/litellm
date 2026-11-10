use std::{borrow::Cow, time::Duration};

use litellm_http::Client;
use litellm_storage_clickhouse::{ClickHouseMigrate, execute_statement, storage_error};
use sqlx::migrate::Migrator;

use crate::{Config, Error};

static MIGRATOR: Migrator = Migrator {
    table_name: Cow::Borrowed("_litellm_spend_migrations"),
    locking: false,
    ..sqlx::migrate!("./migrations")
};

pub async fn ensure_schema(client: &Client, config: &Config) -> Result<(), Error> {
    let storage = config.storage();
    let database = format!("`{}`", storage.database());
    let retention = config.retention_days().to_string();
    let mut adapter = ClickHouseMigrate::new(
        client,
        storage.writer(),
        storage.database(),
        |sql| {
            sql.replace("{database}", &database)
                .replace("{retention_days}", &retention)
        },
        Duration::from_secs(30),
    )?;
    MIGRATOR
        .run_direct(None, &mut adapter, false)
        .await
        .map_err(|error| match storage_error(&error) {
            Some(error) => Error::Storage(error.clone()),
            None => Error::Migration(error),
        })?;
    execute_statement(
        client,
        storage.writer(),
        &format!("ALTER TABLE {database}.spend_logs MODIFY TTL toDateTime(start_time) + INTERVAL {retention} DAY"),
        Duration::from_secs(30),
    ).await?;
    Ok(())
}
