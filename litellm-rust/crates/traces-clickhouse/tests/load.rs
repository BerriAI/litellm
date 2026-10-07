use std::collections::BTreeMap;

use litellm_traces_clickhouse::{Connection, Parameter, ReadQuery, execute_named_read};
use rstest::rstest;
use serde_json::Value;

#[path = "queries/support.rs"]
#[expect(
    dead_code,
    reason = "load tests share the query fixture but do not read through QueryReaders"
)]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database};
use support::TestResult;

const SPANS_PER_DAY: u64 = 2_000;

async fn seed_days(fixture: &SeededDatabase, first_day: u64, days: u64) -> TestResult {
    let count = SPANS_PER_DAY * days;
    let first_row = SPANS_PER_DAY * first_day;
    let query = format!(
        "INSERT INTO {DATABASE}.otel_traces \
         (Timestamp, TraceId, SpanId, ParentSpanId, SpanName, ServiceName, ObservationType, TeamId, ApiKeyHash, Duration) \
         SELECT now64(9) - toIntervalHour(intDiv(number, {SPANS_PER_DAY}) * 24 + 12 + {first_day} * 24), \
         if({first_day} = 0, concat('load-', toString(number + {first_row})), 'load-0'), \
         concat('span-', toString(number + {first_row})), \
         '', 'span', 'service', 'agent', 'load-team', '', 0 FROM numbers({count})"
    );
    fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(query)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

async fn trace_start_time(fixture: &SeededDatabase) -> TestResult<String> {
    let query = format!(
        "SELECT toString(Timestamp, 'UTC') AS start_time FROM {DATABASE}.otel_traces \
         WHERE TraceId = 'load-0' LIMIT 1 FORMAT JSON"
    );
    let response = fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(query)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let result: Value = serde_json::from_str(&response)?;
    result["data"][0]["start_time"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "trace start time missing".into())
}

fn content_parameters(start_time: &str) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        ("source".into(), Parameter::Text("traces".into())),
        ("all_teams".into(), Parameter::Integer(0)),
        ("team".into(), Parameter::Text("load-team".into())),
        ("key_hash".into(), Parameter::Text(String::new())),
        ("id".into(), Parameter::Text("load-0".into())),
        ("record_team".into(), Parameter::Text("load-team".into())),
        ("start_time".into(), Parameter::Text(start_time.into())),
        ("trace_ref".into(), Parameter::Text(String::new())),
        ("cursor".into(), Parameter::Text(String::new())),
        ("offset".into(), Parameter::Integer(1)),
    ])
}

async fn content(fixture: &SeededDatabase, start_time: &str, query_id: &str) -> TestResult {
    let connection = Connection::configured(
        &format!("{}?query_id={query_id}", fixture.database.url),
        DATABASE,
        "default",
        "",
    )?;
    let response = execute_named_read(
        &fixture.database.client,
        &connection,
        ReadQuery::Content,
        &content_parameters(start_time),
    )
    .await?;
    let result: Value = serde_json::from_str(&response)?;
    assert!(!result["data"].as_array().ok_or("content rows")?.is_empty());
    Ok(())
}

async fn query_read_rows(fixture: &SeededDatabase, query_id: &str) -> TestResult<u64> {
    fixture
        .database
        .client
        .post(&fixture.database.url)
        .body("SYSTEM FLUSH LOGS")
        .send()
        .await?
        .error_for_status()?;
    let response = fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(format!(
            "SELECT read_rows FROM system.query_log WHERE type = 'QueryFinish' \
             AND query_id = '{query_id}' ORDER BY event_time DESC LIMIT 1 FORMAT JSON"
        ))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let result: Value = serde_json::from_str(&response)?;
    result["data"][0]["read_rows"]
        .as_u64()
        .ok_or_else(|| "query log read_rows missing".into())
}

#[rstest]
#[tokio::test]
async fn lens_content_reads_scale_with_trace_not_retention(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_days(&fixture, 0, 8).await?;
    let start_time = trace_start_time(&fixture).await?;
    let before_id = format!("lens_content_before_{}", std::process::id());
    content(&fixture, &start_time, &before_id).await?;
    let before = query_read_rows(&fixture, &before_id).await?;

    seed_days(&fixture, 8, 24).await?;
    let after_id = format!("lens_content_after_{}", std::process::id());
    content(&fixture, &start_time, &after_id).await?;
    let after = query_read_rows(&fixture, &after_id).await?;
    println!("lens_content read_rows: before={before}, after={after}");
    assert!(
        after * 100 <= before * 105,
        "read_rows grew from {before} to {after}"
    );
    Ok(())
}
