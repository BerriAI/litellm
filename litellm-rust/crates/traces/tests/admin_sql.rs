use litellm_http::Client;
use litellm_traces::{Connection, Error, execute_admin_sql};
use rstest::{fixture, rstest};
use serde_json::Value;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";

struct Database {
    _container: ContainerAsync<ClickHouse>,
    url: String,
    admin_url: String,
    client: Client,
}

#[fixture]
async fn database() -> Result<Database, Box<dyn std::error::Error>> {
    let container = ClickHouse::default()
        .with_tag(CLICKHOUSE_TAG)
        .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
        .with_env_var("LITELLM_TRACES_READER_PASSWORD", "test_password")
        .with_copy_to(
            "/etc/clickhouse-server/users.d/litellm-traces-reader.xml",
            include_bytes!("../config/reader.xml").to_vec(),
        )
        .start()
        .await?;
    let admin_url = format!(
        "http://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(8123).await?,
    );
    let client = Client::no_redirect_for_test();
    for sql in [
        "CREATE TABLE otel_traces (n UInt8) ENGINE = Memory",
        "INSERT INTO otel_traces VALUES (1)",
        "CREATE TABLE private_traces (n UInt8) ENGINE = Memory",
    ] {
        client
            .post(&admin_url)
            .body(sql)
            .send()
            .await?
            .error_for_status()?;
    }
    let url = admin_url.replacen("http://", "http://litellm_traces_reader:test_password@", 1);
    Ok(Database {
        _container: container,
        url,
        admin_url,
        client,
    })
}

#[rstest]
#[tokio::test]
async fn admin_sql_reads_rows_with_enforced_settings(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let connection = Connection::parse(&format!(
        "{}?readonly=0&default_format=TabSeparated&query=SELECT+2",
        database.url,
    ))?;

    let result = execute_admin_sql(
        &database.client,
        &connection,
        "SELECT n AS answer FROM otel_traces",
    )
    .await?;
    let json: Value = serde_json::from_str(&result)?;
    assert_eq!(json["data"][0]["answer"], 1);

    Ok(())
}

#[rstest]
#[case::table("CREATE TABLE admin_sql_test (n UInt8) ENGINE = Memory")]
#[case::insert("INSERT INTO otel_traces VALUES (2)")]
#[case::drop("DROP TABLE otel_traces")]
#[case::named_collection("CREATE NAMED COLLECTION admin_sql_test AS host = 'localhost'")]
#[case::settings("SET readonly = 0")]
#[case::inline_settings("SELECT n FROM otel_traces SETTINGS readonly = 0")]
#[case::time_limit("SELECT n FROM otel_traces SETTINGS max_execution_time = 0")]
#[case::row_limit("SELECT n FROM otel_traces SETTINGS max_result_rows = 0")]
#[case::byte_limit("SELECT n FROM otel_traces SETTINGS max_result_bytes = 0")]
#[case::memory_limit("SELECT n FROM otel_traces SETTINGS max_memory_usage = 0")]
#[case::other_table("SELECT * FROM private_traces")]
#[tokio::test]
async fn reader_rejects_writes_and_privilege_escalation(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
    #[case] sql: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let connection = Connection::parse(&format!("{}?readonly=0", database.url))?;

    let result = execute_admin_sql(&database.client, &connection, sql).await;

    assert!(matches!(result, Err(Error::QueryFailed(_))), "{result:?}");
    let rows =
        execute_admin_sql(&database.client, &connection, "SELECT n FROM otel_traces").await?;
    let json: Value = serde_json::from_str(&rows)?;
    assert_eq!(json["data"], serde_json::json!([{ "n": 1 }]));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn admin_sql_rejects_errors_after_output_starts(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let connection = Connection::parse(&format!(
        "{}?max_block_size=1&buffer_size=1&http_write_exception_in_output_format=1\
         &send_progress_in_http_headers=1&http_headers_progress_interval_ms=0",
        database.admin_url,
    ))?;

    let result = execute_admin_sql(
        &database.client,
        &connection,
        "SELECT sleepEachRow(0.2), throwIf(number = 2) FROM numbers(5)",
    )
    .await;

    assert!(
        matches!(result, Err(Error::InvalidResponse)),
        "expected an error embedded in a successful HTTP response: {result:?}"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn admin_sql_enforces_result_row_limit(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let connection = Connection::parse(&format!(
        "{}?max_result_rows=0&result_overflow_mode=throw&wait_end_of_query=1",
        database.url,
    ))?;

    let result = execute_admin_sql(
        &database.client,
        &connection,
        "SELECT number FROM numbers(1001)",
    )
    .await;

    assert!(matches!(result, Err(Error::QueryFailed(_))), "{result:?}");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn admin_sql_enforces_response_byte_limit(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    let connection = Connection::parse(&database.admin_url)?;

    let result = execute_admin_sql(
        &database.client,
        &connection,
        "SELECT repeat('x', 5 * 1024 * 1024)",
    )
    .await;

    assert!(matches!(result, Err(Error::ResponseTooLarge)), "{result:?}");
    Ok(())
}

#[rstest]
#[case::plain("test_password", "test_password")]
#[case::encoded("p@ss/word%", "p%40ss%2Fword%25")]
#[tokio::test]
async fn admin_sql_authenticates_url_credentials(
    #[future(awt)] database: Result<Database, Box<dyn std::error::Error>>,
    #[case] password: &str,
    #[case] encoded_password: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let database = database?;
    database
        .client
        .post(&database.admin_url)
        .body(format!(
            "CREATE USER sql_reader IDENTIFIED WITH plaintext_password BY '{password}'"
        ))
        .send()
        .await?
        .error_for_status()?;
    let connection = Connection::parse(&database.admin_url.replacen(
        "http://",
        &format!("http://sql_reader:{encoded_password}@"),
        1,
    ))?;

    let result = execute_admin_sql(
        &database.client,
        &connection,
        "SELECT currentUser() AS username",
    )
    .await?;
    let json: Value = serde_json::from_str(&result)?;

    assert_eq!(json["data"][0]["username"], "sql_reader");

    Ok(())
}
