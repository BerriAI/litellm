use std::{collections::BTreeMap, time::Duration};

use litellm_http::Client;
use litellm_traces::{
    Connection, Error, InsertTable, encode_rows, ensure_schema, execute_read, schema_statements,
};
use rstest::{fixture, rstest};
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ContainerAsync, ImageExt, runners::AsyncRunner},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";

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

async fn insert_rows(
    database: &ClickHouseDatabase,
    table: &str,
    rows: Vec<BTreeMap<String, serde_json::Value>>,
) -> TestResult {
    database
        .client
        .post(&database.url)
        .query(&[
            (
                "query",
                format!("INSERT INTO trace_test.{table} FORMAT JSONEachRow"),
            ),
            ("date_time_input_format", "best_effort".into()),
        ])
        .body(encode_rows(rows)?)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
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
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let body = execute_read(&database.client, &connection, sql, &BTreeMap::new()).await?;
    Ok(serde_json::from_str(&body)?)
}

async fn table_rows(database: &ClickHouseDatabase, table: &str) -> TestResult<u64> {
    let response = read_json(
        database,
        &format!("SELECT count() AS rows FROM trace_test.{table}"),
    )
    .await?;
    Ok(response["data"][0]["rows"]
        .as_u64()
        .expect("ClickHouse returns row counts as unsigned integers"))
}

async fn mutation_rows(database: &ClickHouseDatabase) -> TestResult<u64> {
    let response = read_json(
        database,
        "SELECT count() AS rows FROM system.mutations WHERE database = 'trace_test'",
    )
    .await?;
    Ok(response["data"][0]["rows"]
        .as_u64()
        .expect("ClickHouse returns mutation counts as unsigned integers"))
}

#[rstest]
#[tokio::test]
async fn schema_supports_span_rollups_and_spend_joins(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let span = serde_json::from_value(serde_json::json!({
        "Timestamp": timestamp, "TraceId": "trace-1", "SpanId": "span-1", "ParentSpanId": "",
        "ServiceName": "proxy", "SpanName": "request", "Input": "hello world",
        "ResourceAttributes": {"litellm.team_id": "team-1", "litellm.api_key_hash": "hash-1"},
        "SpanAttributes": {"gen_ai.response.id": "response-1", "gen_ai.usage.input_tokens": "12"}
    }))?;
    let spend = serde_json::from_value(serde_json::json!({
        "request_id": "request-1", "response_id": "response-1", "team_id": "team-1", "spend": 0.125,
        "start_time": timestamp / 1_000_000, "end_time": timestamp / 1_000_000 + 100,
        "completion_start_time": null
    }))?;
    insert_rows(&database, "otel_traces", vec![span]).await?;
    insert_rows(&database, "spend_logs", vec![spend]).await?;
    let body = read_json(
        &database,
        "SELECT o.TeamId, o.ApiKeyHash, o.ObservationType, o.InputPreview, s.spend, \
         toString(toUnixTimestamp64Nano(o.Timestamp)) AS timestamp_ns, \
         toString(toUnixTimestamp64Milli(s.start_time)) AS start_ms \
         FROM trace_test.otel_traces o JOIN trace_test.spend_logs s \
         ON o.LiteLLMRequestId = s.response_id AND o.TeamId = s.team_id",
    )
    .await?;
    assert_eq!(
        body["data"],
        serde_json::json!([{
            "TeamId": "team-1", "ApiKeyHash": "hash-1", "ObservationType": "agent",
            "InputPreview": "hello world", "spend": 0.125,
            "timestamp_ns": timestamp.to_string(), "start_ms": (timestamp / 1_000_000).to_string()
        }])
    );
    let body = read_json(
        &database,
        "SELECT toUInt32(sum(SpanCount)) AS spans, toUInt32(sum(InputTokens)) AS tokens \
         FROM trace_test.agent_traces_by_key WHERE TeamId = 'team-1' AND TraceId = 'trace-1'",
    )
    .await?;
    assert_eq!(
        body["data"],
        serde_json::json!([{"spans": 1, "tokens": 12}])
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn retried_trace_insert_does_not_inflate_rollup(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    let row = serde_json::from_value(serde_json::json!({
        "Timestamp": time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64,
        "TraceId": "retried-trace", "SpanId": "span-1", "ParentSpanId": "",
        "TeamId": "team-1", "ApiKeyHash": "key-1", "SpanName": "root", "InputTokens": 7
    }))?;
    for _ in 0..2 {
        litellm_traces::insert_rows(
            &database.client,
            &writer,
            "trace_test",
            InsertTable::OtelTraces,
            vec![row.clone()],
        )
        .await?;
    }
    let counts = read_json(
        &database,
        "SELECT toUInt32(sum(SpanCount)) AS spans, toUInt32(sum(InputTokens)) AS tokens \
         FROM trace_test.agent_traces_by_key WHERE TraceId = 'retried-trace'",
    )
    .await?;
    assert_eq!(table_rows(&database, "otel_traces").await?, 1);
    assert_eq!(counts["data"][0]["spans"], 1);
    assert_eq!(counts["data"][0]["tokens"], 7);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn keyed_rollup_keeps_same_trace_ids_separate_by_api_key(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let rows = vec![
        serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp, "TraceId": "shared-id", "SpanId": "root-one",
            "ParentSpanId": "", "SpanName": "root-one", "Input": "private-one",
            "ResourceAttributes": {"litellm.api_key_hash": "key-one"}
        }))?,
        serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp, "TraceId": "shared-id", "SpanId": "root-two",
            "ParentSpanId": "", "SpanName": "root-two", "Input": "private-two",
            "ResourceAttributes": {"litellm.api_key_hash": "key-two"}
        }))?,
    ];
    insert_rows(&database, "otel_traces", rows).await?;
    execute_write(
        &database,
        "OPTIMIZE TABLE trace_test.agent_traces_by_key FINAL",
    )
    .await?;
    let rows = read_json(
        &database,
        "SELECT ApiKeyHash, any(RootInput) AS RootInput \
         FROM trace_test.agent_traces_by_key WHERE TraceId = 'shared-id' \
         GROUP BY ApiKeyHash ORDER BY ApiKeyHash",
    )
    .await?;
    assert_eq!(
        rows["data"],
        serde_json::json!([
            {"ApiKeyHash": "key-one", "RootInput": "private-one"},
            {"ApiKeyHash": "key-two", "RootInput": "private-two"}
        ])
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn rollup_merges_spans_across_days_without_losing_root_fields(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    let day_start = time::OffsetDateTime::now_utc()
        .replace_time(time::Time::MIDNIGHT)
        .unix_timestamp_nanos() as i64;
    let root = serde_json::from_value(serde_json::json!({
        "Timestamp": day_start - 1_000_000_000, "TraceId": "cross-day", "SpanId": "span-root",
        "ParentSpanId": "", "ServiceName": "proxy", "SpanName": "root", "Input": "root input",
        "StatusCode": "STATUS_CODE_ERROR",
        "ResourceAttributes": {"litellm.team_id": "team-1"}
    }))?;
    insert_rows(&database, "otel_traces", vec![root]).await?;
    let child = serde_json::from_value(serde_json::json!({
        "Timestamp": day_start + 1_000_000_000, "TraceId": "cross-day", "SpanId": "span-child",
        "ParentSpanId": "span-root", "ServiceName": "proxy", "SpanName": "child",
        "StatusCode": "STATUS_CODE_UNSET",
        "ResourceAttributes": {"litellm.team_id": "team-1"}
    }))?;
    insert_rows(&database, "otel_traces", vec![child]).await?;
    execute_write(
        &database,
        "OPTIMIZE TABLE trace_test.agent_traces_by_key FINAL",
    )
    .await?;
    let response = read_json(
        &database,
        "SELECT count() AS rows, any(RootName) AS RootName, any(RootInput) AS RootInput, \
         any(RootStatus) AS RootStatus, sum(SpanCount) AS SpanCount \
         FROM trace_test.agent_traces_by_key",
    )
    .await?;
    assert_eq!(
        response["data"],
        serde_json::json!([{
            "rows": 1, "RootName": "root", "RootInput": "root input",
            "RootStatus": "STATUS_CODE_ERROR", "SpanCount": 2
        }])
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn spend_deduplication_preserves_subsecond_requests_and_retries(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7, 14).await?;
    let now_ms = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64 / 1_000_000;
    let base_start_time = now_ms / 1000 * 1000;
    let first_start_time = base_start_time + 100;
    let second_start_time = base_start_time + 200;
    let first = serde_json::from_value(serde_json::json!({
        "request_id": "same-request", "team_id": "team-1", "spend": 1.0,
        "start_time": first_start_time, "end_time": first_start_time + 1000
    }))?;
    let second = serde_json::from_value(serde_json::json!({
        "request_id": "same-request", "team_id": "team-1", "spend": 2.0,
        "start_time": second_start_time, "end_time": second_start_time + 1200
    }))?;
    let retry = serde_json::from_value(serde_json::json!({
        "request_id": "same-request", "team_id": "team-1", "spend": 1.0,
        "start_time": first_start_time, "end_time": first_start_time + 2000
    }))?;
    insert_rows(&database, "spend_logs", vec![first]).await?;
    insert_rows(&database, "spend_logs", vec![second]).await?;
    insert_rows(&database, "spend_logs", vec![retry]).await?;
    execute_write(&database, "OPTIMIZE TABLE trace_test.spend_logs FINAL").await?;
    let rows = read_json(
        &database,
        "SELECT toString(toUnixTimestamp64Milli(start_time)) AS start_time, \
         toString(toUnixTimestamp64Milli(end_time)) AS end_time \
         FROM trace_test.spend_logs ORDER BY start_time",
    )
    .await?;
    assert_eq!(
        rows["data"],
        serde_json::json!([
            {
                "start_time": first_start_time.to_string(),
                "end_time": (first_start_time + 2000).to_string()
            },
            {
                "start_time": second_start_time.to_string(),
                "end_time": (second_start_time + 1200).to_string()
            }
        ])
    );
    assert_eq!(table_rows(&database, "spend_logs").await?, 2);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn retention_changes_materialize_existing_rows_and_remain_idempotent(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 30, 30).await?;
    let old_time = time::OffsetDateTime::now_utc() - time::Duration::days(20);
    let old_timestamp_ns = old_time.unix_timestamp_nanos() as i64;
    let old_timestamp_ms = old_timestamp_ns / 1_000_000;
    let span = serde_json::from_value(serde_json::json!({
        "Timestamp": old_timestamp_ns, "TraceId": "expired", "SpanId": "span-old",
        "ParentSpanId": "", "ServiceName": "proxy", "SpanName": "old-root", "Input": "old input",
        "ResourceAttributes": {"litellm.team_id": "team-1"}
    }))?;
    let spend = serde_json::from_value(serde_json::json!({
        "request_id": "old-request", "team_id": "team-1", "spend": 1.0,
        "start_time": old_timestamp_ms, "end_time": old_timestamp_ms + 1000
    }))?;
    insert_rows(&database, "otel_traces", vec![span]).await?;
    insert_rows(&database, "spend_logs", vec![spend]).await?;
    assert_eq!(table_rows(&database, "agent_traces_by_key").await?, 1);
    ensure_schema(&database.client, &writer, "trace_test", 14, 14).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let response = read_json(
            &database,
            "SELECT countIf(is_done = 0) AS pending \
             FROM system.mutations WHERE database = 'trace_test'",
        )
        .await?;
        let pending = response["data"][0]["pending"]
            .as_u64()
            .expect("ClickHouse returns pending mutation counts as unsigned integers");
        if pending == 0 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "ClickHouse TTL mutations did not finish before the deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    execute_write(&database, "OPTIMIZE TABLE trace_test.otel_traces FINAL").await?;
    execute_write(
        &database,
        "OPTIMIZE TABLE trace_test.agent_traces_by_key FINAL",
    )
    .await?;
    execute_write(&database, "OPTIMIZE TABLE trace_test.spend_logs FINAL").await?;
    assert_eq!(table_rows(&database, "otel_traces").await?, 0);
    assert_eq!(table_rows(&database, "agent_traces_by_key").await?, 0);
    assert_eq!(table_rows(&database, "spend_logs").await?, 0);
    let mutation_count = mutation_rows(&database).await?;
    ensure_schema(&database.client, &writer, "trace_test", 14, 14).await?;
    assert_eq!(mutation_rows(&database).await?, mutation_count);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn schema_statement_timeout_maps_to_transport_error() -> TestResult {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (_connection, _) = listener.accept().await.expect("accept schema request");
        std::future::pending::<()>().await;
    });
    let client = Client::no_redirect_for_test();
    let url = format!("http://{address}");
    let writer = Connection::writer(&url)?;
    let result = tokio::time::timeout(
        Duration::from_secs(35),
        ensure_schema(&client, &writer, "trace_test", 7, 14),
    )
    .await;
    server.abort();
    assert!(matches!(result, Ok(Err(Error::Transport))), "{result:?}");
    Ok(())
}

#[rstest]
#[case::empty("", 7, 14)]
#[case::sql("db; DROP DATABASE default", 7, 14)]
#[case::trace_retention("traces", 0, 14)]
#[case::spend_retention("traces", 7, 0)]
fn schema_rejects_invalid_configuration(
    #[case] database: &str,
    #[case] traces: u32,
    #[case] spend: u32,
) {
    assert!(schema_statements(database, traces, spend).is_err());
}
