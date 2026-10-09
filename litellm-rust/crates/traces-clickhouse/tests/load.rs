use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use litellm_storage_clickhouse::READ_LIMITS;
use litellm_traces::query::named::ReadAccessParams;
use litellm_traces_cache::TraceReader;
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, Parameter, ReadQuery, execute_named_read,
};
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
const LONG_TRACE: &str = "long-trace";
const LONG_TRACE_SPANS: u64 = 20_000;
const LIST_TRACES: u64 = 5_000;
const LIST_SPANS_PER_TRACE: u64 = 10;
// read_rows for one 50 row page: 25,848 with the old filter, 10,848 with the optimized filter
const LIST_PAGE_READ_ROWS: u64 = 18_348;

async fn seed_days(fixture: &SeededDatabase, first_day: u64, days: u64) -> TestResult {
    let count = SPANS_PER_DAY * days;
    let first_row = SPANS_PER_DAY * first_day;
    let query = format!(
        "INSERT INTO {DATABASE}.otel_traces \
         (Timestamp, TraceId, SpanId, ParentSpanId, SpanName, ServiceName, ObservationType, TeamId, ApiKeyHash, Duration, SpanAttributes) \
         SELECT now64(9) - toIntervalHour(intDiv(number, {SPANS_PER_DAY}) * 24 + 12 + {first_day} * 24), \
         if({first_day} = 0, concat('load-', toString(number + {first_row})), 'load-0'), \
         concat('span-', toString(number + {first_row})), \
         '', 'span', 'service', 'agent', 'load-team', '', 0, \
         if({first_day}=0 AND number < {SPANS_PER_DAY}, map('payload', repeat('x', 3000)), map()) \
         FROM numbers({count})"
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

fn sample_parameters(start: u64, end: u64) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        ("source".into(), Parameter::Text("traces".into())),
        ("all_teams".into(), Parameter::Integer(0)),
        ("team".into(), Parameter::Text("load-team".into())),
        ("key_hash".into(), Parameter::Text(String::new())),
        ("start".into(), Parameter::Unsigned(start)),
        ("end".into(), Parameter::Unsigned(end)),
        ("agent_name".into(), Parameter::Text(String::new())),
        ("service".into(), Parameter::Text(String::new())),
        ("filter_keys".into(), Parameter::Strings(Vec::new())),
        ("filter_values".into(), Parameter::Strings(Vec::new())),
        ("selected_team".into(), Parameter::Text(String::new())),
        ("execution_ids".into(), Parameter::Strings(Vec::new())),
        ("sample_cap".into(), Parameter::Unsigned(0)),
        ("sample_percent".into(), Parameter::Integer(100)),
        ("preview".into(), Parameter::Integer(0)),
        ("after".into(), Parameter::Text(String::new())),
        ("limit".into(), Parameter::Unsigned(10_000)),
        ("offset".into(), Parameter::Unsigned(0)),
    ])
}

async fn sample(
    fixture: &SeededDatabase,
    start: u64,
    end: u64,
    query_id: &str,
) -> TestResult<(usize, usize)> {
    let mut url = Connection::configured(&fixture.database.url, DATABASE, "default", "")?
        .url()
        .clone();
    url.query_pairs_mut().append_pair("query_id", query_id);
    let connection = Connection::parse(url.as_str())?;
    let response = execute_named_read(
        &fixture.database.client,
        &connection,
        ReadQuery::Sample,
        &sample_parameters(start, end),
    )
    .await?;
    let result: Value = serde_json::from_str(&response)?;
    Ok((
        result["data"].as_array().ok_or("sample rows")?.len(),
        response.len(),
    ))
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

async fn execute(fixture: &SeededDatabase, query: String) -> TestResult<String> {
    Ok(fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(query)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?)
}

fn counter(value: &Value) -> TestResult<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str()?.parse().ok())
        .ok_or_else(|| "query log counter missing".into())
}

async fn seed_long_trace(fixture: &SeededDatabase) -> TestResult {
    execute(
        fixture,
        format!(
            "INSERT INTO {DATABASE}.otel_traces \
             (Timestamp, TraceId, SpanId, ParentSpanId, SpanName, ServiceName, ObservationType, AgentName, TeamId, ApiKeyHash, Duration, SpanAttributes) \
             SELECT now64(9) - toIntervalMillisecond(number), '{LONG_TRACE}', \
             concat('span-', leftPad(toString(number), 6, '0')), \
             if(number = 0, '', 'span-000000'), 'span', 'service', \
             if(number = 0, 'agent', 'tool'), 'load-agent', 'load-team', '', 1000, \
             map('payload', repeat('x', 1000)) FROM numbers({LONG_TRACE_SPANS})"
        ),
    )
    .await?;
    Ok(())
}

async fn seed_trace_pages(fixture: &SeededDatabase) -> TestResult {
    let spans = LIST_TRACES * LIST_SPANS_PER_TRACE;
    execute(
        fixture,
        format!(
            "INSERT INTO {DATABASE}.otel_traces \
             (Timestamp, TraceId, SpanId, ParentSpanId, SpanName, ServiceName, ObservationType, AgentName, Framework, TeamId, ApiKeyHash, UserId, Duration) \
             SELECT now64(9) - toIntervalSecond(intDiv(number, {LIST_SPANS_PER_TRACE})), \
             concat('page-', leftPad(toString(intDiv(number, {LIST_SPANS_PER_TRACE})), 5, '0')), \
             concat('span-', toString(number)), \
             if(number % {LIST_SPANS_PER_TRACE} = 0, '', 'root'), 'span', 'service', \
             if(number % {LIST_SPANS_PER_TRACE} = 0, 'agent', 'llm'), 'load-agent', 'langchain', \
             'load-team', '', 'load-user', 1000 FROM numbers({spans})"
        ),
    )
    .await?;
    Ok(())
}

async fn span_batch_reads(fixture: &SeededDatabase, query_id: &str) -> TestResult<(u64, u64)> {
    execute(fixture, "SYSTEM FLUSH LOGS".into()).await?;
    let response = execute(
        fixture,
        format!(
            "SELECT count() AS batches, sum(read_rows) AS read_rows FROM system.query_log \
             WHERE type = 'QueryFinish' AND current_database = '{DATABASE}' \
             AND query_id = '{query_id}' AND query LIKE '%LIMIT 1 BY o.SpanId%' FORMAT JSON"
        ),
    )
    .await?;
    let result: Value = serde_json::from_str(&response)?;
    Ok((
        counter(&result["data"][0]["batches"])?,
        counter(&result["data"][0]["read_rows"])?,
    ))
}

async fn traces_by_key_rows(fixture: &SeededDatabase) -> TestResult<(u64, u64)> {
    let response = execute(
        fixture,
        format!(
            "SELECT count() AS rows, uniqExact(TraceId) AS traces \
             FROM {DATABASE}.agent_traces_by_key FORMAT JSON"
        ),
    )
    .await?;
    let result: Value = serde_json::from_str(&response)?;
    Ok((
        counter(&result["data"][0]["rows"])?,
        counter(&result["data"][0]["traces"])?,
    ))
}

fn list_parameters(start_ms: u64, end_ms: u64) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text(String::new())),
        (
            "team_ids".into(),
            Parameter::Strings(vec!["load-team".into()]),
        ),
        ("start_ms".into(), Parameter::Unsigned(start_ms)),
        ("end_ms".into(), Parameter::Unsigned(end_ms)),
        ("cursor_ms".into(), Parameter::Unsigned(0)),
        ("cursor_trace_id".into(), Parameter::Text(String::new())),
        ("limit".into(), Parameter::Unsigned(50)),
    ])
}

async fn list_page(
    fixture: &SeededDatabase,
    start_ms: u64,
    end_ms: u64,
    query_id: &str,
) -> TestResult<Value> {
    let mut url = Connection::configured(&fixture.database.url, DATABASE, "default", "")?
        .url()
        .clone();
    url.query_pairs_mut().append_pair("query_id", query_id);
    let connection = Connection::parse(url.as_str())?;
    let response = execute_named_read(
        &fixture.database.client,
        &connection,
        ReadQuery::ListTraces,
        &list_parameters(start_ms, end_ms),
    )
    .await?;
    Ok(serde_json::from_str(&response)?)
}

#[rstest]
#[tokio::test]
async fn lens_sample_reads_scale_with_window_not_retention(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_days(&fixture, 0, 8).await?;
    let now_ms = time::OffsetDateTime::now_utc().unix_timestamp() as u64 * 1000;
    let start = now_ms - 86_400_000;
    let end = now_ms + 60_000;
    let before_id = format!("lens_sample_before_{}", std::process::id());
    let (before_rows, response_bytes) = sample(&fixture, start, end, &before_id).await?;
    assert_eq!(before_rows, SPANS_PER_DAY as usize);
    assert!(response_bytes > READ_LIMITS.response_bytes);
    let before = query_read_rows(&fixture, &before_id).await?;

    seed_days(&fixture, 8, 24).await?;
    let after_id = format!("lens_sample_after_{}", std::process::id());
    let (after_rows, _) = sample(&fixture, start, end, &after_id).await?;
    assert_eq!(after_rows, SPANS_PER_DAY as usize);
    let after = query_read_rows(&fixture, &after_id).await?;
    assert!(
        after * 100 <= before * 105,
        "read_rows grew from {before} to {after}"
    );
    Ok(())
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

#[rstest]
#[tokio::test]
async fn trace_open_reads_a_large_trace_in_few_span_batches(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_long_trace(&fixture).await?;
    let query_id = format!("trace_open_{}", std::process::id());
    let mut reader_url = Connection::configured(&fixture.database.url, DATABASE, "default", "")?
        .url()
        .clone();
    reader_url
        .query_pairs_mut()
        .append_pair("query_id", &query_id);
    let connection = Connection::reader(reader_url.as_str(), DATABASE)?;
    let reader = TraceReader::new(READ_LIMITS.response_bytes);
    let store = ClickHouseTraces::new(fixture.database.client.clone(), connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["load-team".into()],
    };
    let started = Instant::now();
    let trace = reader
        .get_trace(&store, &access, LONG_TRACE, "")
        .await?
        .ok_or("missing trace")?;
    let elapsed = started.elapsed();

    assert_eq!(trace.spans.len(), LONG_TRACE_SPANS as usize);
    let (batches, read_rows) = span_batch_reads(&fixture, &query_id).await?;
    println!("trace_open: batches={batches}, read_rows={read_rows}, elapsed={elapsed:?}");
    assert!(batches <= 4, "{batches} span batch queries");
    assert!(
        read_rows <= 5 * LONG_TRACE_SPANS,
        "{read_rows} rows read for {LONG_TRACE_SPANS} spans"
    );
    assert!(
        elapsed < Duration::from_secs(15),
        "trace read took {elapsed:?}"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn trace_list_page_reads_the_page_once(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_trace_pages(&fixture).await?;
    let (rows, traces) = traces_by_key_rows(&fixture).await?;
    assert_eq!(rows, LIST_TRACES, "expected one aggregate row per trace");
    assert_eq!(traces, LIST_TRACES);

    let now_ms = time::OffsetDateTime::now_utc().unix_timestamp() as u64 * 1000;
    let query_id = format!("trace_list_page_{}", std::process::id());
    let page = list_page(&fixture, now_ms - 86_400_000, now_ms + 60_000, &query_id).await?;
    let data = page["data"].as_array().ok_or("list rows")?;

    assert_eq!(data.len(), 50);
    for run in data {
        assert!(!run["trace_id"].as_str().unwrap_or_default().is_empty());
        assert_eq!(run["agent_names"], serde_json::json!(["load-agent"]));
        assert_eq!(run["agent_count"].as_u64(), Some(1));
    }
    let read_rows = query_read_rows(&fixture, &query_id).await?;
    println!("trace_list_page read_rows={read_rows}");
    assert!(
        read_rows < LIST_PAGE_READ_ROWS,
        "{read_rows} rows read for a 50 row page"
    );
    Ok(())
}
