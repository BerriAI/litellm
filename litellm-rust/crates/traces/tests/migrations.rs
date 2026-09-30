use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_traces::{Connection, execute_read, schema_statements};
use rstest::rstest;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    testcontainers::{ImageExt, runners::AsyncRunner},
};

const CLICKHOUSE_TAG: &str =
    "26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e";

#[rstest]
#[tokio::test]
async fn schema_supports_span_rollups_and_spend_joins() -> Result<(), Box<dyn std::error::Error>> {
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
    let client = Client::no_redirect_for_test();
    for _ in 0..2 {
        for sql in schema_statements("trace_test", 7, 14)? {
            client
                .post(&url)
                .body(sql)
                .send()
                .await?
                .error_for_status()?;
        }
    }
    for sql in [
        "INSERT INTO trace_test.otel_traces \
         (Timestamp, TraceId, SpanId, ParentSpanId, ServiceName, SpanName, ResourceAttributes, SpanAttributes, Input) \
         VALUES (now64(9), 'trace-1', 'span-1', '', 'proxy', 'request', \
         map('litellm.team_id', 'team-1', 'litellm.api_key_hash', 'hash-1'), \
         map('gen_ai.response.id', 'response-1', 'gen_ai.usage.input_tokens', '12'), 'hello world')",
        "INSERT INTO trace_test.spend_logs (request_id, response_id, team_id, spend, start_time, end_time) \
         VALUES ('request-1', 'response-1', 'team-1', 0.125, now64(3), now64(3))",
    ] {
        client
            .post(&url)
            .body(sql)
            .send()
            .await?
            .error_for_status()?;
    }
    let connection = Connection::configured(&url, "trace_test", "default", "")?;
    let body = execute_read(&client, &connection,
        "SELECT o.TeamId, o.ApiKeyHash, o.ObservationType, o.InputPreview, s.spend \
         FROM otel_traces o JOIN spend_logs s ON o.LiteLLMRequestId = s.response_id AND o.TeamId = s.team_id",
        &BTreeMap::new()).await?;
    let response: serde_json::Value = serde_json::from_str(&body)?;
    assert_eq!(
        response["data"],
        serde_json::json!([{
            "TeamId": "team-1", "ApiKeyHash": "hash-1", "ObservationType": "agent",
            "InputPreview": "hello world", "spend": 0.125
        }])
    );
    let body = execute_read(
        &client,
        &connection,
        "SELECT toUInt32(sum(SpanCount)) AS spans, toUInt32(sum(InputTokens)) AS tokens \
         FROM agent_traces WHERE TeamId = 'team-1' AND TraceId = 'trace-1'",
        &BTreeMap::new(),
    )
    .await?;
    let response: serde_json::Value = serde_json::from_str(&body)?;
    assert_eq!(
        response["data"],
        serde_json::json!([{"spans": 1, "tokens": 12}])
    );
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
