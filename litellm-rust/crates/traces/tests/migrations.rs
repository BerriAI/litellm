use litellm_http::Client;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ImageExt, runners::AsyncRunner},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";
const INIT_SQL: &str = include_str!("../migrations/0001_otel_traces.sql");

#[rstest::rstest]
#[tokio::test]
async fn migration_creates_a_queryable_trace_table() -> Result<(), Box<dyn std::error::Error>> {
    let container = ClickHouse::default()
        .with_tag(CLICKHOUSE_TAG)
        .with_env_var("CLICKHOUSE_SKIP_USER_SETUP", "1")
        .start()
        .await?;
    let url = format!(
        "http://{}:{}",
        container.get_host().await?,
        container.get_host_port_ipv4(8123).await?,
    );
    let client = Client::no_redirect_for_test();

    for _ in 0..2 {
        client
            .post(&url)
            .body(INIT_SQL)
            .send()
            .await?
            .error_for_status()?;
    }

    client
        .post(&url)
        .body(
            "INSERT INTO otel_traces \
             (Timestamp, TraceId, SpanId, ParentSpanId, ServiceName, SpanName, SpanAttributes, Input) \
             VALUES (now64(9), 'trace-1', 'span-1', '', 'proxy', 'request', \
             map('litellm.team_id', 'team-1', 'litellm.api_key_hash', 'hash-1'), 'hello world')",
        )
        .send()
        .await?
        .error_for_status()?;

    let row = client
        .post(&url)
        .body(
            "SELECT TeamId, ApiKeyHash, ObservationType, InputPreview \
             FROM otel_traces WHERE TraceId = 'trace-1' FORMAT TSV",
        )
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    assert_eq!(row, "team-1\thash-1\tagent\thello world\n");

    Ok(())
}
