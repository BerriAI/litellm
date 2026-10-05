use std::time::Duration;

use litellm_http::Client;
use litellm_storage_clickhouse::{
    ClickHouseMigrate, Connection, Error, READ_LIMITS, execute_statement, storage_error,
};
use rstest::{fixture, rstest};
use sqlx::{
    SqlStr,
    migrate::{Migrate, MigrateError, Migration, MigrationType, Migrator},
};
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string, method, query_param},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";
const DATABASE: &str = "storage_migrate_test";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const SELECT_APPLIED: &str = "SELECT DISTINCT version, checksum FROM `trace_test`._sqlx_migrations \
                              WHERE success ORDER BY version FORMAT JSONEachRow";

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct ClickHouseDatabase {
    _container: ContainerAsync<ClickHouse>,
    url: String,
    client: Client,
}

#[fixture]
async fn database() -> TestResult<ClickHouseDatabase> {
    let container = ClickHouse::default()
        .with_tag(CLICKHOUSE_TAG)
        .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
        .start()
        .await?;
    let url = format!(
        "http://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(8123).await?
    );
    Ok(ClickHouseDatabase {
        _container: container,
        url,
        client: Client::no_redirect_for_test(),
    })
}

#[fixture]
async fn mock_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .with_priority(10)
        .mount(&server)
        .await;
    server
}

fn migration(version: i64, sql: &'static str) -> Migration {
    Migration::new(
        version,
        format!("migration_{version}").into(),
        MigrationType::Simple,
        SqlStr::from_static(sql),
        false,
    )
}

fn migrator(migrations: Vec<Migration>) -> Migrator {
    Migrator {
        ignore_missing: true,
        locking: false,
        ..Migrator::with_migrations(migrations)
    }
}

fn render_database(sql: &str) -> String {
    sql.replace("{database}", &format!("`{DATABASE}`"))
}

async fn run_migrations<R>(
    database: &ClickHouseDatabase,
    migrator: &Migrator,
    schema: &str,
    render: R,
) -> Result<(), MigrateError>
where
    R: Fn(&str) -> String + Send + Sync,
{
    let connection = Connection::writer(&database.url).expect("valid ClickHouse URL");
    let mut adapter = ClickHouseMigrate::new(
        &database.client,
        &connection,
        schema,
        render,
        REQUEST_TIMEOUT,
    )
    .expect("valid schema");
    migrator.run_direct(None, &mut adapter, false).await
}

async fn execute_write(database: &ClickHouseDatabase, sql: &str) -> TestResult {
    database
        .client
        .post(&database.url)
        .body(sql.to_owned())
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

async fn read_json(database: &ClickHouseDatabase, sql: &str) -> TestResult<serde_json::Value> {
    let response = database
        .client
        .post(&database.url)
        .body(sql.to_owned())
        .send()
        .await?
        .error_for_status()?;
    Ok(serde_json::from_str(&response.text().await?)?)
}

async fn ledger_versions(database: &ClickHouseDatabase) -> TestResult<Vec<i64>> {
    let response = read_json(
        database,
        &format!(
            "SELECT version FROM `{DATABASE}`._sqlx_migrations \
             GROUP BY version ORDER BY version FORMAT JSON"
        ),
    )
    .await?;
    Ok(response["data"]
        .as_array()
        .expect("ClickHouse returns versions")
        .iter()
        .map(|row| row["version"].as_i64().expect("version is Int64"))
        .collect())
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[rstest]
#[tokio::test]
async fn only_pending_migrations_execute_on_the_second_run(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let migrator = migrator(vec![
        migration(
            1,
            "CREATE TABLE IF NOT EXISTS {database}.migration_one (id UInt8) ENGINE = MergeTree ORDER BY id",
        ),
        migration(
            2,
            "CREATE TABLE IF NOT EXISTS {database}.migration_two (id UInt8) ENGINE = MergeTree ORDER BY id",
        ),
    ]);
    run_migrations(&database, &migrator, DATABASE, render_database).await?;
    run_migrations(&database, &migrator, DATABASE, render_database).await?;
    execute_statement(
        &database.client,
        &Connection::writer(&database.url)?,
        "SYSTEM FLUSH LOGS",
        REQUEST_TIMEOUT,
    )
    .await?;

    let queries = read_json(
        &database,
        "SELECT count() AS executions FROM system.query_log \
         WHERE type = 'QueryFinish' AND query LIKE \
         'CREATE TABLE IF NOT EXISTS `storage_migrate_test`.migration_%' FORMAT JSON",
    )
    .await?;
    assert_eq!(queries["data"][0]["executions"].as_u64(), Some(2));
    assert_eq!(ledger_versions(&database).await?, vec![1, 2]);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn edited_migration_checksum_returns_version_mismatch(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let original = migrator(vec![migration(
        1,
        "CREATE TABLE IF NOT EXISTS {database}.original (id UInt8) ENGINE = MergeTree ORDER BY id",
    )]);
    let changed = migrator(vec![migration(
        1,
        "CREATE TABLE IF NOT EXISTS {database}.changed (id UInt8) ENGINE = MergeTree ORDER BY id",
    )]);
    run_migrations(&database, &original, DATABASE, render_database).await?;

    assert!(matches!(
        run_migrations(&database, &changed, DATABASE, render_database).await,
        Err(MigrateError::VersionMismatch(1))
    ));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn failed_migration_is_not_recorded_and_retains_storage_error(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let migrator = migrator(vec![migration(1, "THIS IS NOT VALID CLICKHOUSE SQL")]);
    let error = run_migrations(&database, &migrator, DATABASE, render_database)
        .await
        .expect_err("invalid SQL must fail");
    assert!(matches!(&error, MigrateError::ExecuteMigration(_, 1)));
    assert!(matches!(
        storage_error(&error),
        Some(Error::SchemaFailed(_))
    ));

    let rows = read_json(
        &database,
        &format!(
            "SELECT count() AS rows FROM `{DATABASE}`._sqlx_migrations \
             WHERE version = 1 FORMAT JSON"
        ),
    )
    .await?;
    assert_eq!(rows["data"][0]["rows"].as_u64(), Some(0));
    Ok(())
}

#[rstest]
fn invalid_database_identifier_is_rejected() {
    let client = Client::no_redirect_for_test();
    let connection = Connection::writer("http://127.0.0.1:1").expect("valid URL");
    assert!(matches!(
        ClickHouseMigrate::new(
            &client,
            &connection,
            "storage_test; DROP DATABASE default",
            str::to_owned,
            REQUEST_TIMEOUT,
        ),
        Err(Error::InvalidSchema)
    ));
}

#[rstest]
#[tokio::test]
async fn unknown_source_version_is_tolerated(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    run_migrations(&database, &migrator(vec![]), DATABASE, render_database).await?;
    execute_write(
        &database,
        &format!(
            "INSERT INTO `{DATABASE}`._sqlx_migrations \
             (version, description, success, checksum, execution_time) \
             VALUES (99, 'unknown', true, '{}', 0)",
            "00".repeat(48)
        ),
    )
    .await?;
    let migrator = migrator(vec![migration(
        1,
        "CREATE TABLE IF NOT EXISTS {database}.known (id UInt8) ENGINE = MergeTree ORDER BY id",
    )]);

    run_migrations(&database, &migrator, DATABASE, render_database).await?;

    assert_eq!(ledger_versions(&database).await?, vec![1, 99]);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn duplicate_ledger_rows_are_tolerated(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let applied = migration(
        1,
        "CREATE TABLE IF NOT EXISTS {database}.duplicate_test (id UInt8) ENGINE = MergeTree ORDER BY id",
    );
    let checksum = encode_hex(&applied.checksum);
    let migrator = migrator(vec![applied]);
    run_migrations(&database, &migrator, DATABASE, render_database).await?;
    execute_write(
        &database,
        &format!(
            "INSERT INTO `{DATABASE}`._sqlx_migrations \
             (version, description, success, checksum, execution_time) \
             VALUES (1, 'migration_1', true, '{checksum}', 0)"
        ),
    )
    .await?;

    run_migrations(&database, &migrator, DATABASE, render_database).await?;

    let rows = read_json(
        &database,
        &format!(
            "SELECT count() AS rows, uniqExact(version) AS versions \
             FROM `{DATABASE}`._sqlx_migrations FORMAT JSON"
        ),
    )
    .await?;
    assert_eq!(rows["data"][0]["rows"].as_u64(), Some(2));
    assert_eq!(rows["data"][0]["versions"].as_u64(), Some(1));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn revert_reports_forward_only_error() {
    let client = Client::no_redirect_for_test();
    let connection = Connection::writer("http://127.0.0.1:1").expect("valid URL");
    let migration = migration(
        1,
        "CREATE TABLE IF NOT EXISTS {database}.revert_test (id UInt8) ENGINE = MergeTree ORDER BY id",
    );
    let mut adapter = ClickHouseMigrate::new(
        &client,
        &connection,
        DATABASE,
        render_database,
        REQUEST_TIMEOUT,
    )
    .expect("valid schema");

    let error = adapter
        .revert("_sqlx_migrations", &migration)
        .await
        .expect_err("ClickHouse migrations cannot be reverted");
    assert!(
        error
            .to_string()
            .contains("ClickHouse migrations are forward-only")
    );
}

#[rstest]
#[case::numeric("1")]
#[case::quoted("\"1\"")]
#[tokio::test]
async fn applied_int64_versions_accept_numeric_and_quoted_json(
    #[future(awt)] mock_server: MockServer,
    #[case] version: &str,
) {
    Mock::given(method("POST"))
        .and(body_string(SELECT_APPLIED))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(format!("{{\"version\":{version},\"checksum\":\"00\"}}\n")),
        )
        .mount(&mock_server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::writer(&mock_server.uri()).expect("valid URL");
    let migrator = migrator(vec![]);
    let mut adapter = ClickHouseMigrate::new(
        &client,
        &connection,
        "trace_test",
        str::to_owned,
        REQUEST_TIMEOUT,
    )
    .expect("valid schema");

    migrator
        .run_direct(None, &mut adapter, false)
        .await
        .expect("applied version parses");
}

#[rstest]
#[tokio::test]
async fn schema_requests_override_unsafe_connection_settings(
    #[future(awt)] mock_server: MockServer,
) {
    Mock::given(method("POST"))
        .and(query_param("wait_end_of_query", "1"))
        .and(query_param("send_progress_in_http_headers", "0"))
        .and(query_param("async_insert", "0"))
        .and(query_param("custom_setting", "preserved"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&mock_server)
        .await;
    let url = format!(
        "{}/?wait_end_of_query=0&send_progress_in_http_headers=1&async_insert=1&custom_setting=preserved",
        mock_server.uri()
    );
    execute_statement(
        &Client::no_redirect_for_test(),
        &Connection::writer(&url).expect("valid URL"),
        "CREATE DATABASE IF NOT EXISTS trace_test",
        REQUEST_TIMEOUT,
    )
    .await
    .expect("schema execution succeeds");
    let requests = mock_server
        .received_requests()
        .await
        .expect("requests recorded");
    for name in [
        "wait_end_of_query",
        "send_progress_in_http_headers",
        "async_insert",
    ] {
        assert_eq!(
            requests[0]
                .url
                .query_pairs()
                .filter(|(key, _)| key == name)
                .count(),
            1
        );
    }
}

#[rstest]
#[tokio::test]
async fn oversized_ledger_response_is_rejected_before_migrations(
    #[future(awt)] mock_server: MockServer,
) {
    Mock::given(method("POST"))
        .and(body_string(SELECT_APPLIED))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(" ".repeat(READ_LIMITS.response_bytes + 1)),
        )
        .mount(&mock_server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::writer(&mock_server.uri()).expect("valid URL");
    let migrator = migrator(vec![]);
    let mut adapter = ClickHouseMigrate::new(
        &client,
        &connection,
        "trace_test",
        str::to_owned,
        REQUEST_TIMEOUT,
    )
    .expect("valid schema");
    let error = migrator
        .run_direct(None, &mut adapter, false)
        .await
        .expect_err("oversized result is rejected");

    assert!(matches!(
        storage_error(&error),
        Some(Error::ResponseTooLarge)
    ));
    assert_eq!(
        mock_server
            .received_requests()
            .await
            .expect("requests recorded")
            .len(),
        3
    );
}
