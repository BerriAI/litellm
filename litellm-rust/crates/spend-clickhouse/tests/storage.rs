use std::{collections::BTreeMap, time::Duration};

use litellm_http::Client;
use litellm_spend_clickhouse::{Config, Error, ensure_schema, insert_rows};
use litellm_storage_clickhouse::{execute_read, execute_statement};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};
use time::OffsetDateTime;

struct Database {
    _container: Option<ContainerAsync<ClickHouse>>,
    config: Config,
    client: Client,
}

impl Database {
    async fn execute(&self, sql: &str) {
        execute_statement(
            &self.client,
            self.config.storage().writer(),
            sql,
            Duration::from_secs(30),
        )
        .await
        .unwrap();
    }

    async fn read(&self, sql: &str) -> Value {
        serde_json::from_str(
            &execute_read(
                &self.client,
                self.config.storage().reader(),
                sql,
                &BTreeMap::new(),
            )
            .await
            .unwrap(),
        )
        .unwrap()
    }

    async fn close(self) {
        self.execute(&format!(
            "DROP DATABASE `{}`",
            self.config.storage().database()
        ))
        .await;
    }
}

#[fixture]
async fn database() -> Database {
    let (container, url) = match std::env::var("CLICKHOUSE_SPEND_TEST_URL") {
        Ok(url) => (None, url),
        Err(_) => {
            let container = ClickHouse::default()
                .with_tag("26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e")
                .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
                .start().await.unwrap();
            let url = format!(
                "http://{}:{}",
                container.get_host().await.unwrap(),
                container.get_host_port_ipv4(8123).await.unwrap()
            );
            (Some(container), url)
        }
    };
    Database {
        _container: container,
        config: Config::new(
            format!("spend_test_{}", uuid::Uuid::new_v4().simple()),
            &url,
            7,
        )
        .unwrap(),
        client: Client::no_redirect_for_test(),
    }
}

#[rstest]
#[case::unknown(Value::Null)]
#[case::free(json!(0))]
#[case::paid(json!(0.125))]
#[tokio::test]
async fn spend_rows_preserve_fields_and_retries(
    #[future(awt)] database: Database,
    #[case] spend: Value,
) {
    ensure_schema(&database.client, &database.config)
        .await
        .unwrap();
    let storage = database.config.storage();
    let start = (OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64;
    let row = BTreeMap::from([
        ("request_id".into(), json!("request-1")),
        ("response_id".into(), json!("response-1")),
        ("provider_request_id".into(), json!("provider-1")),
        ("litellm_call_id".into(), json!("call-1")),
        ("team_id".into(), json!("team-1")),
        ("api_key".into(), json!("key-hash")),
        ("user".into(), json!("user-1")),
        ("start_time".into(), json!(start)),
        ("end_time".into(), json!(start + 1234)),
        ("completion_start_time".into(), Value::Null),
        ("spend".into(), spend.clone()),
        ("prompt_tokens".into(), json!(17)),
        ("completion_tokens".into(), json!(5)),
        ("metadata".into(), json!("{\"message\":\"雪\"}")),
        ("request_tags".into(), json!(["tag-1", "tag-2"])),
        ("EngineReceivedMs".into(), json!(0)),
    ]);
    insert_rows(
        &database.client,
        storage.writer(),
        storage.database(),
        vec![row.clone()],
    )
    .await
    .unwrap();
    insert_rows(
        &database.client,
        storage.writer(),
        storage.database(),
        vec![row],
    )
    .await
    .unwrap();
    let read = database.read("SELECT request_id, response_id, provider_request_id, litellm_call_id, team_id, api_key, user, toString(toUnixTimestamp64Milli(start_time)) AS start_ms, toString(toUnixTimestamp64Milli(end_time)) AS end_ms, completion_start_time, spend, prompt_tokens, completion_tokens, metadata, request_tags, EngineReceivedMs > 0 AS received FROM spend_logs FINAL").await;
    assert_eq!(
        read["data"],
        json!([{
            "request_id":"request-1", "response_id":"response-1", "provider_request_id":"provider-1", "litellm_call_id":"call-1",
            "team_id":"team-1", "api_key":"key-hash", "user":"user-1", "start_ms":start.to_string(), "end_ms":(start+1234).to_string(),
            "completion_start_time":null, "spend":spend, "prompt_tokens":17, "completion_tokens":5,
            "metadata":"{\"message\":\"雪\"}", "request_tags":["tag-1","tag-2"], "received":1
        }])
    );
    database.close().await;
}

#[rstest]
#[tokio::test]
async fn spend_schema_preserves_existing_tables_and_reconciles_retention(
    #[future(awt)] database: Database,
) {
    let storage = database.config.storage();
    let name = storage.database();
    database.execute(&format!("CREATE DATABASE `{name}`")).await;
    database
        .execute(&format!(
            "CREATE TABLE `{name}`.otel_traces (marker String) ENGINE=MergeTree ORDER BY marker"
        ))
        .await;
    database
        .execute(&format!(
            "INSERT INTO `{name}`.otel_traces VALUES ('preserved')"
        ))
        .await;
    database.execute(&format!("CREATE TABLE `{name}`._sqlx_migrations (version Int64, checksum String) ENGINE=MergeTree ORDER BY version")).await;
    database
        .execute(&format!(
            "INSERT INTO `{name}`._sqlx_migrations VALUES (6, 'other-product-checksum')"
        ))
        .await;
    database
        .execute(
            &include_str!("../migrations/0006_spend_logs.sql")
                .replace("{database}", &format!("`{name}`")),
        )
        .await;
    database.execute(&format!("INSERT INTO `{name}`.spend_logs (request_id, start_time, end_time, spend) VALUES ('legacy-spend', now64(3), now64(3), 0.125)")).await;
    ensure_schema(&database.client, &database.config)
        .await
        .unwrap();
    let updated = Config::new(name.into(), storage.writer().url().as_str(), 14).unwrap();
    ensure_schema(&database.client, &updated).await.unwrap();
    ensure_schema(&database.client, &updated).await.unwrap();
    let rows = database.read("SELECT marker FROM otel_traces").await;
    assert_eq!(rows["data"], json!([{"marker":"preserved"}]));
    assert_eq!(
        database
            .read("SELECT request_id, spend FROM spend_logs FINAL")
            .await["data"],
        json!([{"request_id":"legacy-spend","spend":0.125}])
    );
    assert_eq!(
        database.read("SELECT * FROM _sqlx_migrations").await["data"],
        json!([{"version":6,"checksum":"other-product-checksum"}])
    );
    let ddl = database.read(&format!("SELECT create_table_query FROM system.tables WHERE database = '{name}' AND name = 'spend_logs'")).await;
    assert!(
        ddl["data"][0]["create_table_query"]
            .as_str()
            .unwrap()
            .contains("toIntervalDay(14)")
    );
    let applied = database.read("SELECT version, count() AS copies FROM _litellm_spend_migrations GROUP BY version ORDER BY version").await;
    assert_eq!(
        applied["data"],
        json!([
            {"version":6,"copies":1},{"version":7,"copies":1},
            {"version":14,"copies":1},{"version":15,"copies":1},{"version":16,"copies":1}
        ])
    );
    database.close().await;
}

#[rstest]
#[tokio::test]
async fn fresh_spend_schema_does_not_create_lens_tables(#[future(awt)] database: Database) {
    ensure_schema(&database.client, &database.config)
        .await
        .unwrap();
    let tables = database.read("SHOW TABLES").await;
    assert_eq!(
        tables["data"],
        json!([{"name":"_litellm_spend_migrations"},{"name":"spend_logs"}])
    );
    database.close().await;
}

#[rstest]
#[tokio::test]
async fn changed_applied_spend_migration_is_rejected(#[future(awt)] database: Database) {
    ensure_schema(&database.client, &database.config)
        .await
        .unwrap();
    database.execute(&format!("ALTER TABLE `{}`._litellm_spend_migrations UPDATE checksum = repeat('00', 48) WHERE version = 6 SETTINGS mutations_sync = 2", database.config.storage().database())).await;
    assert!(matches!(
        ensure_schema(&database.client, &database.config).await,
        Err(Error::Migration(
            sqlx::migrate::MigrateError::VersionMismatch(6)
        ))
    ));
    database.close().await;
}

#[rstest]
#[tokio::test]
async fn unexpected_columns_are_rejected_without_writing_rows(#[future(awt)] database: Database) {
    ensure_schema(&database.client, &database.config)
        .await
        .unwrap();
    let storage = database.config.storage();
    let result = insert_rows(
        &database.client,
        storage.writer(),
        storage.database(),
        vec![BTreeMap::from([
            ("request_id".into(), json!("invalid")),
            ("not_a_spend_column".into(), json!("value")),
        ])],
    )
    .await;
    assert!(matches!(
        result,
        Err(Error::Storage(
            litellm_storage_clickhouse::Error::InsertFailed(_)
        ))
    ));
    assert_eq!(
        database
            .read("SELECT count() AS rows FROM spend_logs")
            .await["data"],
        json!([{"rows":0}])
    );
    database.close().await;
}
