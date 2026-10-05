use std::{collections::BTreeMap, time::Duration};

use litellm_http::Client;
use litellm_traces::{
    search::RunFilter,
    store::{
        CallQuery, RunCursor, RunQuery, RunSelection, SpanPart, SpanQuery, SpanSelection, TextRange,
    },
};
use litellm_traces_cache::{TraceReader, TraceStore};
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, Error, InsertTable, NORMALIZED_FIELD_DEFINITIONS, QueryScope,
    encode_rows, ensure_schema, execute_read, schema_statements,
};
use rstest::rstest;
use sha2::{Digest, Sha256};
mod support;

use support::{ClickHouseDatabase, TestResult, database};

fn owned(user_id: &str, team_ids: &[&str]) -> QueryScope {
    QueryScope::Owned {
        user_id: user_id.into(),
        team_ids: team_ids.iter().map(|team| (*team).to_owned()).collect(),
    }
}

fn traces(database: &ClickHouseDatabase, connection: &Connection) -> ClickHouseTraces {
    ClickHouseTraces::new(database.client.clone(), connection.clone())
}

fn trace_ref(team_id: &str, api_key_hash: &str, trace_id: &str) -> String {
    format!(
        "{:X}",
        Sha256::digest(format!("{team_id}\0{api_key_hash}\0{trace_id}"))
    )
}

async fn list_runs(
    database: &ClickHouseDatabase,
    connection: &Connection,
    access: &QueryScope,
    window: std::ops::Range<i64>,
    after: Option<RunCursor>,
    limit: u32,
) -> TestResult<serde_json::Value> {
    let query = RunQuery {
        order: Default::default(),
        selection: RunSelection::Matching(RunFilter {
            start_ms: window.start,
            end_ms: window.end,
            search: Default::default(),
            ..Default::default()
        }),
        after,
        limit,
    };
    let rows = traces(database, connection).runs(access, &query).await?;
    Ok(serde_json::json!({ "data": rows }))
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
async fn schema_supports_trace_summaries_and_spend_joins(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let span = serde_json::from_value(serde_json::json!({
        "Timestamp": timestamp, "TraceId": "trace-1", "SpanId": "span-1", "ParentSpanId": "",
        "ServiceName": "proxy", "SpanName": "request", "Input": "hello world",
        "ResourceAttributes": {"litellm.team_id": "team-1", "litellm.api_key_hash": "hash-1", "litellm.user_id": "exporter-claim"},
        "SpanAttributes": {"gen_ai.response.id": "response-1", "gen_ai.usage.input_tokens": "12"}
    }))?;
    let spend = serde_json::from_value(serde_json::json!({
        "request_id": "request-1", "response_id": "response-1", "team_id": "team-1", "spend": 0.125,
        "start_time": timestamp / 1_000_000, "end_time": timestamp / 1_000_000 + 100,
        "completion_start_time": null
    }))?;
    insert_rows(&database, "otel_traces", vec![span]).await?;
    insert_rows(&database, "spend_logs", vec![spend]).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let team = owned("", &["team-1"]);
    let detail = TraceReader::new(usize::MAX)
        .get_span(&traces(&database, &reader), &team, "trace-1", "span-1", "")
        .await?
        .ok_or("missing span detail")?;
    assert_eq!(detail.input, "hello world");
    assert_eq!(detail.attributes["gen_ai.response.id"], "response-1");
    let calls = traces(&database, &reader)
        .calls(
            &team,
            &CallQuery {
                as_of_ms: u64::MAX,
                window: timestamp / 1_000_000 - 1000..timestamp / 1_000_000 + 1000,
                response_ids: vec!["response-1".into()],
                request_ids: Vec::new(),
                trace_ids: Vec::new(),
                after: None,
                limit: 10,
            },
        )
        .await?;
    assert_eq!(calls[0].spend, Some(0.125));
    let body = read_json(
        &database,
        "SELECT o.TeamId, o.ApiKeyHash, o.UserId, o.ObservationType, o.InputPreview, s.spend, \
         toString(toUnixTimestamp64Nano(o.Timestamp)) AS timestamp_ns, \
         toString(toUnixTimestamp64Milli(s.start_time)) AS start_ms \
         FROM trace_test.otel_traces o JOIN trace_test.spend_logs s \
         ON o.LiteLLMRequestId = s.response_id AND o.TeamId = s.team_id",
    )
    .await?;
    assert_eq!(
        body["data"],
        serde_json::json!([{
            "TeamId": "team-1", "ApiKeyHash": "hash-1", "UserId": "", "ObservationType": "agent",
            "InputPreview": "hello world", "spend": 0.125,
            "timestamp_ns": timestamp.to_string(), "start_ms": (timestamp / 1_000_000).to_string()
        }])
    );
    let body = read_json(
        &database,
        "SELECT toUInt32(span_count) AS spans, toUInt32(span_input_tokens) AS tokens \
         FROM trace_test.traces WHERE team_id = 'team-1' AND trace_id = 'trace-1'",
    )
    .await?;
    assert_eq!(
        body["data"],
        serde_json::json!([{"spans": 1, "tokens": 12}])
    );
    let ledger = read_json(
        &database,
        "SELECT toUInt32(count()) AS applied, toUInt32(uniqExact(version)) AS versions \
         FROM trace_test.schema_migrations",
    )
    .await?;
    let files = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/migrations"))?.count();
    assert_eq!(
        ledger["data"],
        serde_json::json!([{"applied": files, "versions": files}]),
        "each migration is recorded once even though the schema was ensured twice"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn normalized_fields_match_clickhouse_catalog(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    ensure_schema(
        &database.client,
        &Connection::writer(&database.url)?,
        "trace_test",
        7,
    )
    .await?;
    let catalog = read_json(&database, "SELECT name, type FROM system.columns WHERE database = 'trace_test' AND table = 'otel_traces'").await?;
    let columns: BTreeMap<&str, &str> = catalog["data"]
        .as_array()
        .expect("catalog rows")
        .iter()
        .map(|row| {
            (
                row["name"].as_str().expect("column name"),
                row["type"].as_str().expect("column type"),
            )
        })
        .collect();
    for field in NORMALIZED_FIELD_DEFINITIONS {
        assert_eq!(
            columns.get(field.clickhouse_column).copied(),
            Some(field.clickhouse_type),
            "{}",
            field.name
        );
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn agent_metadata_is_stored_and_queryable(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let ready = database?;
    ensure_schema(
        &ready.client,
        &Connection::writer(&ready.url)?,
        "trace_test",
        7,
    )
    .await?;
    let metadata = serde_json::json!({"thread_id": "thread-1", "ls_subagent_id": "agent-1"});
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    insert_rows(
        &ready,
        "otel_traces",
        vec![BTreeMap::from([
            ("Timestamp".into(), timestamp.into()),
            ("TraceId".into(), "trace-1".into()),
            ("SpanId".into(), "span-1".into()),
            ("AgentMetadata".into(), metadata.to_string().into()),
        ])],
    )
    .await?;
    let response = read_json(
        &ready,
        "SELECT JSONExtractString(AgentMetadata, 'thread_id') AS thread_id, JSONExtractString(AgentMetadata, 'ls_subagent_id') AS subagent_id FROM trace_test.otel_traces WHERE TraceId = 'trace-1'",
    ).await?;
    assert_eq!(response["data"][0]["thread_id"], metadata["thread_id"]);
    assert_eq!(
        response["data"][0]["subagent_id"],
        metadata["ls_subagent_id"]
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn insert_rejects_unknown_columns_even_if_url_requests_skipping_them(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&format!(
        "{}?input_format_skip_unknown_fields=1",
        database.url
    ))?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let row = BTreeMap::from([
        (
            "Timestamp".to_owned(),
            serde_json::json!(1_700_000_000_000_000_000_i64),
        ),
        (
            "unexpected".to_owned(),
            serde_json::json!("dropped silently"),
        ),
    ]);

    assert!(matches!(
        litellm_traces_clickhouse::insert_rows(
            &database.client,
            &writer,
            "trace_test",
            InsertTable::OtelTraces,
            vec![row]
        )
        .await,
        Err(Error::Storage(
            litellm_storage_clickhouse::Error::InsertFailed(_)
        ))
    ));
    assert_eq!(table_rows(&database, "otel_traces").await?, 0);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn retried_trace_insert_does_not_inflate_span_rows(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let row: BTreeMap<String, serde_json::Value> = serde_json::from_value(serde_json::json!({
        "Timestamp": time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64,
        "TraceId": "retried-trace", "SpanId": "span-1", "ParentSpanId": "",
        "TeamId": "team-1", "ApiKeyHash": "key-1", "SpanName": "root", "InputTokens": 7
    }))?;
    for _ in 0..2 {
        litellm_traces_clickhouse::insert_rows(
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
        "SELECT toUInt32(span_count) AS spans, toUInt32(span_input_tokens) AS tokens \
         FROM trace_test.traces WHERE trace_id = 'retried-trace'",
    )
    .await?;
    assert_eq!(table_rows(&database, "otel_traces").await?, 1);
    assert_eq!(table_rows(&database, "spans_core").await?, 1);
    assert_eq!(counts["data"][0]["spans"], 1);
    assert_eq!(counts["data"][0]["tokens"], 7);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn keyed_summaries_keep_same_trace_ids_separate_by_api_key(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
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
    let rows = read_json(
        &database,
        "SELECT api_key_hash AS ApiKeyHash, input_preview AS RootInput \
         FROM trace_test.traces WHERE trace_id = 'shared-id' ORDER BY api_key_hash",
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
async fn listed_agent_names_preserve_scope_and_cursor(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    for (team, key, trace, agent, span, parent, framework) in [
        (
            "alpha",
            "one",
            "shared",
            "research_agent",
            "root",
            "",
            "claude-code",
        ),
        (
            "alpha",
            "one",
            "shared",
            "reviewer",
            "child",
            "root",
            "claude-agent-sdk",
        ),
        (
            "alpha",
            "one",
            "shared",
            "reviewer",
            "repeated",
            "root",
            "claude-agent-sdk",
        ),
        ("alpha", "one", "shared", "", "unnamed", "root", ""),
        ("alpha", "one", "second", "support_agent", "root", "", ""),
        (
            "alpha",
            "two",
            "shared",
            "private_agent",
            "root",
            "",
            "private-sdk",
        ),
        (
            "beta",
            "other",
            "shared",
            "other_agent",
            "root",
            "",
            "other-sdk",
        ),
    ] {
        insert_rows(
            &database,
            "otel_traces",
            vec![serde_json::from_value(serde_json::json!({
                "Timestamp": timestamp, "TraceId": trace, "SpanId": span, "ParentSpanId": parent,
                "ServiceName": "shared-app", "SpanName": span, "AgentName": agent,
                "UserId": if key == "one" { "owner" } else { "other" },
                "Framework": framework, "ObservationType": "agent",
                "ResourceAttributes": {"litellm.team_id": team, "litellm.api_key_hash": key}
            }))?],
        )
        .await?;
    }
    let historical_rows = (0..5000)
        .map(|index| {
            serde_json::from_value(serde_json::json!({
                "Timestamp": timestamp - 86_400_000_000_000_i64,
                "TraceId": "shared", "SpanId": format!("historical-{index}"),
                "ParentSpanId": "", "SpanName": "historical", "AgentName": "private_agent",
                "ObservationType": "agent", "ServiceName": "shared-app",
                "ResourceAttributes": {"litellm.team_id": "alpha", "litellm.api_key_hash": "history"}
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    insert_rows(&database, "otel_traces", historical_rows).await?;
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let owner = owned("owner", &[]);
    let window = timestamp / 1_000_000 - 1000..timestamp / 1_000_000 + 1000;
    let first = list_runs(&database, &connection, &owner, window.clone(), None, 1).await?;
    let after = RunCursor {
        value: first["data"][0]["start_ms"]
            .as_i64()
            .ok_or("missing start")?,
        trace_ref: first["data"][0]["trace_ref"]
            .as_str()
            .ok_or("missing cursor")?
            .into(),
    };
    let second = list_runs(&database, &connection, &owner, window, Some(after), 1).await?;
    assert_eq!(
        first["data"].as_array().ok_or("missing first page")?.len(),
        1
    );
    assert_eq!(
        second["data"]
            .as_array()
            .ok_or("missing second page")?
            .len(),
        1
    );
    assert_ne!(first["data"][0]["trace_id"], second["data"][0]["trace_id"]);
    let names = [&first["data"][0], &second["data"][0]]
        .into_iter()
        .map(|row| {
            (
                row["trace_id"].as_str().unwrap(),
                row["agent_names"].clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        names["shared"],
        serde_json::json!(["research_agent", "reviewer", "unnamed"])
    );
    assert_eq!(names["second"], serde_json::json!(["support_agent"]));
    let frameworks = [&first["data"][0], &second["data"][0]]
        .into_iter()
        .map(|row| (row["trace_id"].as_str().unwrap(), row["frameworks"].clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        frameworks["shared"],
        serde_json::json!(["claude-agent-sdk", "claude-code"])
    );
    assert_eq!(frameworks["second"], serde_json::json!([]));
    let counts = [&first["data"][0], &second["data"][0]]
        .into_iter()
        .map(|row| {
            (
                row["trace_id"].as_str().unwrap(),
                row["agent_count"].as_u64(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(counts["shared"], Some(3));
    assert_eq!(counts["second"], Some(1));
    execute_write(&database, "SYSTEM FLUSH LOGS").await?;
    let reads = read_json(
        &database,
        "SELECT count() AS pages, max(read_rows) AS rows FROM system.query_log \
         WHERE type = 'QueryFinish' AND current_database = 'trace_test' \
         AND position(query, 'page AS (') > 0 AND query NOT LIKE '%system.query_log%'",
    )
    .await?;
    assert_eq!(reads["data"][0]["pages"], 2);
    assert!(
        reads["data"][0]["rows"]
            .as_u64()
            .ok_or("missing read statistics")?
            < 5000
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn a_run_is_summarized_across_days_when_its_root_arrives_last(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let day_start = time::OffsetDateTime::now_utc()
        .replace_time(time::Time::MIDNIGHT)
        .unix_timestamp_nanos() as i64;
    let root = serde_json::from_value(serde_json::json!({
        "Timestamp": day_start - 1_000_000_000, "TraceId": "cross-day", "SpanId": "span-root",
        "ParentSpanId": "", "ServiceName": "proxy", "SpanName": "root", "Input": "root input",
        "AgentName": "lead", "ObservationType": "agent",
        "StatusCode": "STATUS_CODE_ERROR",
        "ResourceAttributes": {"litellm.team_id": "team-1"}
    }))?;
    let child = serde_json::from_value(serde_json::json!({
        "Timestamp": day_start + 1_000_000_000, "TraceId": "cross-day", "SpanId": "span-child",
        "ParentSpanId": "span-root", "ServiceName": "proxy", "SpanName": "child",
        "AgentName": "researcher", "ObservationType": "agent",
        "StatusCode": "STATUS_CODE_UNSET",
        "ResourceAttributes": {"litellm.team_id": "team-1"}
    }))?;
    insert_rows(&database, "otel_traces", vec![child]).await?;
    insert_rows(&database, "otel_traces", vec![root]).await?;
    let response = read_json(
        &database,
        "SELECT name AS Name, input_preview AS RootInput, root_status AS RootStatus, \
         toUInt32(span_count) AS SpanCount FROM trace_test.traces",
    )
    .await?;
    assert_eq!(
        response["data"],
        serde_json::json!([{
            "Name": "root", "RootInput": "root input",
            "RootStatus": "error", "SpanCount": 2
        }])
    );
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let listed = list_runs(
        &database,
        &connection,
        &owned("", &["team-1"]),
        day_start / 1_000_000 - 2000..day_start / 1_000_000,
        None,
        10,
    )
    .await?;
    assert_eq!(listed["data"].as_array().map(Vec::len), Some(1));
    assert_eq!(listed["data"][0]["start_ms"], day_start / 1_000_000 - 1000);
    assert_eq!(
        listed["data"][0]["agent_names"],
        serde_json::json!(["lead", "researcher"])
    );
    assert_eq!(listed["data"][0]["agent_count"], 2);
    let after_start = list_runs(
        &database,
        &connection,
        &owned("", &["team-1"]),
        day_start / 1_000_000..day_start / 1_000_000 + 2000,
        None,
        10,
    )
    .await?;
    assert_eq!(after_start["data"], serde_json::json!([]));
    Ok(())
}

#[rstest]
#[tokio::test]
async fn spend_deduplication_preserves_subsecond_requests_and_retries(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
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
    ensure_schema(&database.client, &writer, "trace_test", 30).await?;
    let tables = read_json(
        &database,
        "SELECT name FROM system.tables WHERE database = 'trace_test' \
         AND match(engine_full, 'materialize_ttl_recalculate_only = 1') ORDER BY name",
    )
    .await?;
    assert_eq!(
        tables["data"],
        serde_json::json!([
            {"name": "otel_traces"},
            {"name": "spans_core"},
            {"name": "spend_logs"}
        ])
    );
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
    assert_eq!(table_rows(&database, "spans_core").await?, 1);
    ensure_schema(&database.client, &writer, "trace_test", 14).await?;
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
    execute_write(&database, "OPTIMIZE TABLE trace_test.spans_core FINAL").await?;
    execute_write(&database, "OPTIMIZE TABLE trace_test.spend_logs FINAL").await?;
    assert_eq!(table_rows(&database, "otel_traces").await?, 0);
    assert_eq!(table_rows(&database, "spans_core").await?, 0);
    assert_eq!(table_rows(&database, "spend_logs").await?, 0);
    let mutation_count = mutation_rows(&database).await?;
    ensure_schema(&database.client, &writer, "trace_test", 14).await?;
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
        ensure_schema(&client, &writer, "trace_test", 7),
    )
    .await;
    server.abort();
    assert!(
        matches!(result, Ok(Err(Error::SchemaTransport))),
        "{result:?}"
    );
    Ok(())
}

#[rstest]
#[case::empty("", 7)]
#[case::sql("db; DROP DATABASE default", 7)]
#[case::retention("traces", 0)]
fn schema_rejects_invalid_configuration(#[case] database: &str, #[case] retention_days: u32) {
    assert!(schema_statements(database, retention_days).is_err());
}

#[rstest]
#[tokio::test]
async fn reused_trace_ids_stay_separate_runs_through_filters_and_span_text(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    for (key, text) in [("one", "timeout"), ("two", "success")] {
        insert_rows(&database, "otel_traces", vec![serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp, "TraceId": "shared", "SpanId": "root", "ParentSpanId": "",
            "ServiceName": "review", "SpanName": "release", "Input": text, "UserId": key,
            "ResourceAttributes": {"litellm.team_id": "team", "litellm.api_key_hash": key, "swarm": "release"}
        }))?]).await?;
    }
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let store = traces(&database, &connection);
    let window = timestamp / 1_000_000 - 1000..timestamp / 1_000_000 + 1000;
    let matched = list_runs(
        &database,
        &connection,
        &QueryScope::All,
        window.clone(),
        None,
        10,
    )
    .await?;
    let filtered = store
        .runs(
            &QueryScope::All,
            &RunQuery {
                order: Default::default(),
                selection: RunSelection::Matching(RunFilter {
                    start_ms: window.start,
                    end_ms: window.end,
                    search: litellm_traces::search::RunSearch::parse(
                        "service:review attr.swarm:release",
                    )
                    .unwrap(),
                    ..Default::default()
                }),
                after: None,
                limit: 10,
            },
        )
        .await?;
    assert_eq!(filtered.len(), 2);
    assert_eq!(matched["data"].as_array().map(Vec::len), Some(2));
    assert_ne!(filtered[0].trace_ref, filtered[1].trace_ref);
    let by_trace_id = RunQuery {
        order: Default::default(),
        selection: RunSelection::TraceId("shared".into()),
        after: None,
        limit: 10,
    };
    assert_eq!(
        store.runs(&owned("", &["team"]), &by_trace_id).await?.len(),
        2
    );
    let own = store.runs(&owned("one", &[]), &by_trace_id).await?;
    assert_eq!(own.len(), 1);
    let reader = TraceReader::new(usize::MAX);
    for run in &filtered {
        let metadata = reader
            .get_trace_metadata(&store, &QueryScope::All, &run.trace_ref)
            .await?
            .unwrap();
        assert_eq!(metadata.summary.trace_ref, run.trace_ref);
        assert_eq!(metadata.summary.input_preview, run.input_preview);
        let spans = reader
            .get_trace_spans(&store, &QueryScope::All, &run.trace_ref, None, 10)
            .await?
            .unwrap();
        assert_eq!(spans.data.len(), 1);
        assert_eq!(spans.data[0].input_preview, run.input_preview);
    }
    assert!(
        reader
            .get_trace_metadata(&store, &owned("two", &[]), &own[0].trace_ref)
            .await?
            .is_none()
    );
    let read = |trace_ref: String, contains: &'static str| {
        let reader = &reader;
        let store = &store;
        async move {
            reader
                .span_text(
                    store,
                    &QueryScope::All,
                    "shared",
                    &trace_ref,
                    vec!["root".into()],
                    SpanPart::Input,
                    TextRange::ALL,
                    Some(contains.into()),
                )
                .await
        }
    };
    let texts = read(own[0].trace_ref.clone(), "timeout").await?;
    assert_eq!(
        texts
            .iter()
            .map(|text| (text.text.as_str(), text.contains))
            .collect::<Vec<_>>(),
        [("timeout", true)]
    );
    let other = read(own[0].trace_ref.clone(), "success").await?;
    assert!(other.iter().all(|text| !text.contains));
    Ok(())
}

#[rstest]
#[case::ascii(10, format!("ParentCommand: {}", "x".repeat(460_000)))]
#[case::multibyte(1_000, "\u{1f9ea}".repeat(1_024))]
#[case::escaped(1_000, "\0\n\"\\".repeat(1_024))]
#[tokio::test]
async fn trace_error_previews_preserve_paginated_diagnostics(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] span_count: usize,
    #[case] message: String,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let rows = (0..span_count)
        .map(|index| {
            serde_json::from_value(serde_json::json!({
                "Timestamp": timestamp + index as i64, "TraceId": "diagnostic-trace",
                "SpanId": format!("span-{index}"), "SpanName": "tool",
                "StatusCode": "STATUS_CODE_ERROR", "StatusMessage": message,
            }))
        })
        .collect::<Result<Vec<BTreeMap<String, serde_json::Value>>, _>>()?;
    insert_rows(&database, "otel_traces", rows).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let store = traces(&database, &reader);
    let reference = trace_ref("", "", "diagnostic-trace");
    let spans = store
        .spans(
            &QueryScope::All,
            &SpanQuery {
                selection: SpanSelection::Trace {
                    trace_id: "diagnostic-trace".into(),
                    trace_ref: reference.clone(),
                },
                as_of_ms: u64::MAX,
                after: None,
                limit: 1000,
            },
        )
        .await?;
    assert_eq!(spans.len(), span_count);
    let prefix: String = message.chars().take(128).collect();
    assert!(!prefix.is_empty());
    assert!(
        spans
            .iter()
            .all(|span| span.status_message == prefix && span.error_truncated)
    );
    let reader = TraceReader::new(usize::MAX);
    let mut recovered = String::new();
    let mut cursor = None;
    loop {
        let page = reader
            .get_span_error(
                &store,
                &QueryScope::All,
                "diagnostic-trace",
                "span-0",
                &reference,
                cursor.as_deref(),
            )
            .await?
            .ok_or("missing diagnostic page")?;
        assert!(!page.message.is_empty());
        assert!(page.message.chars().count() <= 16_384);
        recovered.push_str(&page.message);
        let Some(next) = page.next_cursor else {
            break;
        };
        cursor = Some(next);
    }
    assert_eq!(recovered, message);
    let denied = reader
        .get_span_error(
            &store,
            &owned("unrelated-user", &[]),
            "diagnostic-trace",
            "span-0",
            &reference,
            None,
        )
        .await?;
    assert!(denied.is_none());
    Ok(())
}

#[rstest]
#[case::different_start(1, 0)]
#[case::different_receive(0, 1)]
#[case::tied_timestamps(0, 0)]
#[tokio::test]
async fn duplicate_span_preview_matches_diagnostic(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] start_delta: i64,
    #[case] receive_delta: i64,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let message = "a".repeat(200);
    let rows = [
        (start_delta, receive_delta, "z".repeat(200)),
        (0, 0, message.clone()),
    ]
    .into_iter()
    .map(|(start_delta, receive_delta, message)| {
        serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp + start_delta, "EngineReceivedMs": 100 + receive_delta,
            "TraceId": "duplicate-trace", "SpanId": "duplicate-span", "StatusMessage": message,
        }))
    })
    .collect::<Result<Vec<BTreeMap<String, serde_json::Value>>, _>>()?;
    insert_rows(&database, "otel_traces", rows).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let store = traces(&database, &reader);
    let reference = trace_ref("", "", "duplicate-trace");
    let preview = store
        .spans(
            &QueryScope::All,
            &SpanQuery {
                selection: SpanSelection::Trace {
                    trace_id: "duplicate-trace".into(),
                    trace_ref: reference.clone(),
                },
                as_of_ms: u64::MAX,
                after: None,
                limit: 10,
            },
        )
        .await?;
    let diagnostic = TraceReader::new(usize::MAX)
        .get_span_error(
            &store,
            &QueryScope::All,
            "duplicate-trace",
            "duplicate-span",
            &reference,
            None,
        )
        .await?
        .ok_or("missing diagnostic")?;
    assert_eq!(preview.len(), 1);
    assert_eq!(preview[0].status_message, message[..128]);
    assert_eq!(diagnostic.message, message);
    Ok(())
}

#[rstest]
fn schema_includes_every_migration_file() -> TestResult {
    let files = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/migrations"))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "sql"))
        .count();
    assert_eq!(schema_statements("trace_test", 7)?.len(), 1 + files);
    Ok(())
}

#[rstest]
#[case::empty(false)]
#[case::custom_metadata(true)]
#[tokio::test]
async fn query_help_discovers_live_schema_and_runs_its_examples(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] populated: bool,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    execute_write(&database, "CREATE USER help_reader").await?;
    for table in [
        "traces",
        "spans",
        "calls",
        "otel_traces",
        "spans_core",
        "spend_logs",
    ] {
        execute_write(
            &database,
            &format!("GRANT SELECT ON trace_test.{table} TO help_reader"),
        )
        .await?;
    }
    let reader = Connection::configured(&database.url, "trace_test", "help_reader", "")?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    if populated {
        execute_write(&database, "SYSTEM STOP MERGES trace_test.spend_logs").await?;
        insert_rows(
            &database,
            "spend_logs",
            vec![serde_json::from_value(serde_json::json!({
                "request_id": "request-1", "response_id": "response-1", "team_id": "team-1",
                "api_key": "key-1", "metadata": r#"{"obsolete":true,"labels":{"priority":"old"}}"#,
                "start_time": timestamp / 1_000_000, "end_time": timestamp / 1_000_000
            }))?],
        )
        .await?;
        let metadata = serde_json::json!({
            "project": "example", "labels": {"priority": 3, "enabled": true},
            "dotted.key": "private-metadata-value", "quote'\\key": null, "items": [{"name": "first"}],
            "<custom>&{{key}}": {"nested.key": true}
        });
        insert_rows(
            &database,
            "spend_logs",
            vec![serde_json::from_value(serde_json::json!({
                "request_id": "request-1", "response_id": "response-1", "team_id": "team-1",
                "api_key": "key-1", "trace_id": "trace-1", "metadata": metadata.to_string(), "spend": 0.25,
                "start_time": timestamp / 1_000_000, "end_time": timestamp / 1_000_000 + 100
            }))?],
        )
        .await?;
        insert_rows(
            &database,
            "otel_traces",
            vec![serde_json::from_value(serde_json::json!({
                "Timestamp": timestamp, "TraceId": "trace-1", "SpanId": "span-1",
                "TeamId": "team-1", "ApiKeyHash": "key-1", "ObservationType": "llm",
                "LiteLLMRequestId": "response-1", "SpanAttributes": {"custom.tag": "value"},
                "ResourceAttributes": {"custom.resource": "value"}
            }))?],
        )
        .await?;
        execute_write(
            &database,
            "ALTER TABLE trace_test.otel_traces ADD COLUMN CustomColumn String",
        )
        .await?;
    }
    let help = serde_json::to_value(
        litellm_traces_clickhouse::query_help(&database.client, &reader).await?,
    )?;
    let keys: std::collections::BTreeSet<_> = help
        .as_object()
        .ok_or("missing help object")?
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        std::collections::BTreeSet::from([
            "access",
            "attributes",
            "dialect",
            "examples",
            "gotchas",
            "guide",
            "metadata",
            "normalized_fields",
            "relationships",
            "response",
            "tables",
        ])
    );
    let guide = help["guide"].as_str().ok_or("missing rendered guide")?;
    assert!(guide.starts_with("Trace SQL query guide"));
    for table in [
        "traces",
        "spans",
        "calls",
        "otel_traces",
        "spans_core",
        "spend_logs",
    ] {
        let described = read_json(&database, &format!("DESCRIBE TABLE {table}")).await?;
        let schema = help["tables"]
            .as_array()
            .ok_or("missing tables")?
            .iter()
            .find(|schema| schema["name"] == table)
            .ok_or("missing table")?;
        assert_eq!(schema["columns"], described["data"]);
        for column in described["data"].as_array().ok_or("missing live columns")? {
            assert!(guide.contains(&format!(
                "{}: {}",
                column["name"].as_str().ok_or("column name")?,
                column["type"].as_str().ok_or("column type")?
            )));
        }
    }
    let gotchas = help["gotchas"].as_array().ok_or("missing gotchas")?;
    let gotcha_positions = gotchas
        .iter()
        .map(|gotcha| {
            guide
                .find(gotcha.as_str().expect("gotcha text"))
                .expect("rendered gotcha")
        })
        .collect::<Vec<_>>();
    assert!(gotcha_positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        guide.contains(
            help["metadata"]["sample_sql"]
                .as_str()
                .ok_or("sampling SQL")?
        )
    );
    for catalog in help["attributes"].as_array().ok_or("attributes")? {
        assert!(guide.contains(catalog["discovery_sql"].as_str().ok_or("discovery SQL")?));
        assert!(guide.contains(&format!("Truncated: {}", catalog["truncated"])));
    }
    assert_eq!(
        guide.contains("No attribute keys found in the sampled spans"),
        !populated
    );
    let tables = help["tables"].as_array().ok_or("missing tables")?;
    assert_eq!(tables.len(), 6);
    let columns = tables
        .iter()
        .find(|table| table["name"] == "otel_traces")
        .ok_or("missing raw span table")?["columns"]
        .as_array()
        .ok_or("missing columns")?;
    for field in NORMALIZED_FIELD_DEFINITIONS {
        assert!(
            columns
                .iter()
                .any(|column| column["name"] == field.clickhouse_column
                    && column["type"] == field.clickhouse_type)
        );
        assert!(
            help["normalized_fields"]
                .as_array()
                .ok_or("missing mappings")?
                .iter()
                .any(|mapped| {
                    mapped["name"] == field.name && mapped["column"] == field.clickhouse_column
                })
        );
    }
    let fields = help["metadata"]["fields"]
        .as_array()
        .ok_or("missing metadata fields")?;
    assert_eq!(fields.is_empty(), !populated);
    assert_eq!(help["metadata"]["truncated"], false);
    assert!(guide.contains(help["metadata"]["scope"].as_str().ok_or("missing scope")?));
    assert_eq!(
        guide.contains("No metadata paths found in the sampled rows"),
        !populated
    );
    if populated {
        let versions = read_json(&database, "SELECT count() AS count FROM spend_logs").await?;
        assert_eq!(versions["data"][0]["count"], 2);
        assert_eq!(help["metadata"]["sampled_rows"], 1);
        assert!(
            !fields
                .iter()
                .any(|field| field["path"] == serde_json::json!(["obsolete"]))
        );
        assert!(
            columns
                .iter()
                .any(|column| column["name"] == "CustomColumn")
        );
        assert!(fields.iter().any(|field| field["path"]
            == serde_json::json!(["labels", "priority"])
            && field["types"] == serde_json::json!(["integer"])));
        assert!(
            fields
                .iter()
                .any(|field| field["path"] == serde_json::json!(["items", 1, "name"]))
        );
        assert!(guide.contains("CustomColumn: String"));
        assert!(!guide.contains("private-metadata-value"));
        assert!(guide.contains("JSONExtractRaw(metadata, '<custom>&{{key}}', 'nested.key')"));
        assert!(guide.contains("span_attributes['custom.tag']"));
        assert!(guide.contains("resource_attributes['custom.resource']"));
        assert_eq!(help["attributes"][0]["fields"][0]["key"], "custom.tag");
        assert_eq!(help["attributes"][1]["fields"][0]["key"], "custom.resource");
        for field in fields {
            let expression = field["expression"].as_str().ok_or("missing expression")?;
            assert!(
                guide.contains(expression),
                "missing plain-text expression: {expression}"
            );
            let sql = format!("SELECT {expression} AS value FROM spend_logs FINAL");
            let body =
                litellm_traces_clickhouse::query_sql(&database.client, &reader, &sql).await?;
            let values: serde_json::Value = serde_json::from_str(&body)?;
            assert_ne!(values["data"][0]["value"], "");
        }
    }
    let examples = help["examples"].as_array().ok_or("missing examples")?;
    let example_positions = examples
        .iter()
        .map(|example| {
            let rendered = format!(
                "{}\n{}",
                example["name"].as_str().expect("name"),
                example["sql"].as_str().expect("SQL")
            );
            guide.find(&rendered).expect("rendered example")
        })
        .collect::<Vec<_>>();
    assert!(example_positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        example_positions.last().ok_or("last example")?
            < gotcha_positions.first().ok_or("first gotcha")?
    );
    for example in examples {
        let sql = example["sql"].as_str().ok_or("missing example SQL")?;
        assert!(guide.contains(example["name"].as_str().ok_or("missing example name")?));
        assert!(guide.contains(sql));
        assert_eq!(
            example
                .as_object()
                .ok_or("example object")?
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from(["name", "sql"])
        );
        let body = litellm_traces_clickhouse::query_sql(&database.client, &reader, sql).await?;
        let values: serde_json::Value = serde_json::from_str(&body)?;
        assert_eq!(
            values["data"].as_array().ok_or("missing data")?.is_empty(),
            !populated
                || matches!(
                    example["name"].as_str(),
                    Some(
                        "LLM spans without a direct spend match"
                            | "Recent failed spans"
                            | "Filter calls by nested metadata"
                    )
                ),
            "{sql}"
        );
        if populated && example["name"] == "Traces correlated with LLM call metadata" {
            assert_eq!(values["data"][0]["trace_id"], "trace-1");
            assert_eq!(values["data"][0]["spend"], 0.25);
        }
    }
    Ok(())
}

#[rstest]
#[case::metadata(2, 1)]
#[case::attributes(1, 2)]
#[case::all(2, 2)]
#[tokio::test]
async fn query_help_preserves_schema_and_guide_when_discovery_hits_reader_limits(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] spend_rows: usize,
    #[case] span_rows: usize,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    execute_write(
        &database,
        "CREATE USER help_reader SETTINGS max_rows_to_read = 1",
    )
    .await?;
    for table in [
        "traces",
        "spans",
        "calls",
        "otel_traces",
        "spans_core",
        "spend_logs",
    ] {
        execute_write(
            &database,
            &format!("GRANT SELECT ON trace_test.{table} TO help_reader"),
        )
        .await?;
    }
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let spend = (0..spend_rows)
        .map(|index| {
            serde_json::from_value(serde_json::json!({
                "request_id": format!("request-{index}"), "start_time": timestamp / 1_000_000,
                "end_time": timestamp / 1_000_000, "metadata": r#"{"custom":{"enabled":true}}"#
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    insert_rows(&database, "spend_logs", spend).await?;
    let spans = (0..span_rows).map(|index| serde_json::from_value(serde_json::json!({
        "Timestamp": timestamp, "TraceId": "trace", "SpanId": format!("span-{index}"),
        "SpanAttributes": {"custom.span": "value"}, "ResourceAttributes": {"custom.resource": "value"}
    }))).collect::<Result<Vec<_>, _>>()?;
    insert_rows(&database, "otel_traces", spans).await?;
    let reader = Connection::configured(&database.url, "trace_test", "help_reader", "")?;
    let help = serde_json::to_value(
        litellm_traces_clickhouse::query_help(&database.client, &reader).await?,
    )?;
    assert_eq!(help["tables"].as_array().ok_or("tables")?.len(), 6);
    assert!(!help["examples"].as_array().ok_or("examples")?.is_empty());
    assert_eq!(
        help["normalized_fields"]
            .as_array()
            .ok_or("normalized fields")?
            .len(),
        NORMALIZED_FIELD_DEFINITIONS.len()
    );
    let guide = help["guide"].as_str().ok_or("guide")?;
    assert!(guide.contains("TraceId: String"));
    assert_eq!(
        guide.contains("Metadata discovery unavailable:"),
        spend_rows > 1
    );
    assert_eq!(
        guide.contains("Attribute discovery unavailable:"),
        span_rows > 1
    );
    assert!(!guide.contains("No metadata paths found in the sampled rows"));
    assert!(!guide.contains("No attribute keys found in the sampled spans"));
    assert!(
        guide.contains(
            help["metadata"]["sample_sql"]
                .as_str()
                .ok_or("sampling SQL")?
        )
    );
    for catalog in help["attributes"].as_array().ok_or("attributes")? {
        assert!(guide.contains(catalog["discovery_sql"].as_str().ok_or("discovery SQL")?));
    }
    for (catalog, unavailable) in [
        (&help["metadata"], spend_rows > 1),
        (&help["attributes"][0], span_rows > 1),
        (&help["attributes"][1], span_rows > 1),
    ] {
        assert_eq!(catalog.get("error").is_some(), unavailable);
        assert_eq!(catalog["truncated"], unavailable);
        assert_eq!(
            catalog["fields"].as_array().ok_or("fields")?.is_empty(),
            unavailable
        );
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn query_help_displays_discovery_truncation(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    execute_write(
        &database,
        "INSERT INTO trace_test.otel_traces (Timestamp, TraceId, SpanId, SpanAttributes, ResourceAttributes) \
         SELECT now64(9), 'trace', 'span', \
         mapFromArrays(arrayMap(x -> concat('key-', toString(x)), range(1000)), arrayMap(x -> 'value', range(1000))) AS attributes, \
         attributes FROM numbers(1)",
    )
    .await?;
    execute_write(
        &database,
        "INSERT INTO trace_test.spend_logs (request_id, start_time, end_time, metadata) \
         SELECT toString(number), now64(3), now64(3), '{\"key\":true}' FROM numbers(1000)",
    )
    .await?;
    let reader = Connection::configured(&database.url, "trace_test", "default", "")?;
    let help = serde_json::to_value(
        litellm_traces_clickhouse::query_help(&database.client, &reader).await?,
    )?;
    let guide = help["guide"].as_str().ok_or("guide")?;
    assert_eq!(help["metadata"]["truncated"], true);
    assert!(guide.contains("truncated: true"));
    for catalog in help["attributes"].as_array().ok_or("attributes")? {
        assert_eq!(catalog["truncated"], true);
        let displayed = format!(
            "{}.{}",
            catalog["table"].as_str().ok_or("table")?,
            catalog["column"].as_str().ok_or("column")?
        );
        let section = guide.split(&displayed).nth(1).ok_or("attribute section")?;
        assert!(
            section
                .split("\n\n")
                .next()
                .ok_or("catalog body")?
                .contains("Truncated: true")
        );
        for field in catalog["fields"].as_array().ok_or("fields")? {
            assert!(section.contains(field["expression"].as_str().ok_or("expression")?));
        }
    }
    Ok(())
}

#[rstest]
fn field_definitions_match_serialized_normalized_span() {
    use std::collections::BTreeSet;

    use litellm_traces::{Tenant, decode_otlp};
    use litellm_traces_clickhouse::span_rows;
    let spans = decode_otlp(
        br#"{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"11111111111111111111111111111111","spanId":"2222222222222222","name":"root"}]}]}]}"#,
        Some("application/json"),
    )
    .expect("valid OTLP");
    let tenant = Tenant {
        team_id: "team".into(),
        api_key_hash: "key".into(),
        ..Tenant::default()
    };
    let rows = span_rows(spans, &tenant, 64 * 1024);
    let row =
        serde_json::to_value(rows.first().expect("storage row")).expect("serializable storage row");
    let keys: BTreeSet<_> = row
        .as_object()
        .expect("storage row object")
        .keys()
        .map(String::as_str)
        .collect();
    let mapped: BTreeSet<_> = NORMALIZED_FIELD_DEFINITIONS
        .iter()
        .map(|field| field.clickhouse_column)
        .collect();
    assert!(mapped.is_subset(&keys));
}

#[rstest]
#[case::own_user("owner", vec![], vec!["own"])]
#[case::own_user_and_permitted_team("owner", vec!["permitted"], vec!["own", "team"])]
#[case::no_identity("", vec![], vec![])]
#[tokio::test]
async fn trusted_and_sql_readers_share_request_log_visibility(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] user: &str,
    #[case] teams: Vec<&str>,
    #[case] expected: Vec<&str>,
) -> TestResult {
    use litellm_traces_clickhouse::QueryReaders;

    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let rows = [("own", "unpermitted", "owner", "request-key"), ("team", "permitted", "other", "other-key"), ("foreign", "foreign", "other", "foreign-key")]
        .into_iter()
        .map(|(id, team, owner, api_key)| serde_json::from_value(serde_json::json!({
            "request_id": id, "response_id": "shared-response", "team_id": team, "user": owner,
            "api_key": api_key, "spend": 0.25, "start_time": timestamp / 1_000_000, "end_time": timestamp / 1_000_000,
        })))
        .collect::<Result<Vec<BTreeMap<String, serde_json::Value>>, _>>()?;
    insert_rows(&database, "spend_logs", rows).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let spend = traces(&database, &reader)
        .calls(
            &owned(user, &teams),
            &CallQuery {
                as_of_ms: u64::MAX,
                window: timestamp / 1_000_000 - 1..timestamp / 1_000_000 + 1,
                response_ids: vec!["shared-response".into()],
                request_ids: Vec::new(),
                trace_ids: Vec::new(),
                after: None,
                limit: 10,
            },
        )
        .await?;
    let actual: std::collections::BTreeSet<_> =
        spend.iter().map(|row| row.request_id.as_str()).collect();
    let expected: std::collections::BTreeSet<_> = expected.into_iter().collect();
    assert_eq!(actual, expected);
    let scope = QueryScope::Owned {
        user_id: user.into(),
        team_ids: teams.into_iter().map(str::to_owned).collect(),
    };
    if user.is_empty() && scope.validate().is_err() {
        assert!(
            QueryReaders::new(writer, "trace_test".into())
                .connection(&database.client, &scope, "secret")
                .await
                .is_err()
        );
        return Ok(());
    }
    let scoped = QueryReaders::new(writer, "trace_test".into())
        .connection(&database.client, &scope, "secret")
        .await?;
    let result: serde_json::Value = serde_json::from_str(
        &litellm_traces_clickhouse::query_sql(
            &database.client,
            &scoped,
            "SELECT request_id FROM spend_logs FINAL ORDER BY request_id",
        )
        .await?,
    )?;
    assert_eq!(
        result["data"],
        serde_json::json!(
            expected
                .into_iter()
                .map(|id| serde_json::json!({"request_id": id}))
                .collect::<Vec<_>>()
        )
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn listed_runs_include_a_user_only_when_the_user_wrote_every_span(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let initial_mutations = mutation_rows(&database).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let span = |index: usize, trace: &str, kind: &str, user: &str| {
        serde_json::from_value::<BTreeMap<String, serde_json::Value>>(serde_json::json!({
            "Timestamp": timestamp, "TraceId": trace, "SpanId": index.to_string(), "TeamId": "team",
            "ApiKeyHash": "export", "UserId": user, "ObservationType": kind,
        }))
    };
    insert_rows(
        &database,
        "otel_traces",
        vec![
            span(0, "complete", "llm", "owner")?,
            span(1, "complete", "llm", "owner")?,
            span(2, "complete", "agent", "owner")?,
            span(3, "mixed", "llm", "owner")?,
        ],
    )
    .await?;
    insert_rows(
        &database,
        "otel_traces",
        vec![span(4, "mixed", "llm", "other")?],
    )
    .await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let window = timestamp / 1_000_000 - 1..timestamp / 1_000_000 + 1;
    let listed = list_runs(
        &database,
        &reader,
        &owned("", &["team"]),
        window.clone(),
        None,
        10,
    )
    .await?;
    let listed = listed["data"].as_array().ok_or("missing runs")?;
    assert_eq!(listed.len(), 2);
    for row in listed {
        match row["trace_id"].as_str().ok_or("missing trace id")? {
            "complete" => {
                assert_eq!(row["user_id"], "owner");
                assert_eq!(row["llm_calls"], 2);
            }
            "mixed" => assert_eq!(row["user_id"], ""),
            id => panic!("unexpected trace {id}"),
        }
    }
    let owner = list_runs(&database, &reader, &owned("owner", &[]), window, None, 10).await?;
    let owner: Vec<_> = owner["data"]
        .as_array()
        .ok_or("missing owned runs")?
        .iter()
        .filter_map(|row| row["trace_id"].as_str())
        .collect();
    assert_eq!(owner, ["complete"]);
    assert_eq!(mutation_rows(&database).await?, initial_mutations);
    Ok(())
}

#[rstest]
#[case::admin(QueryScope::All, Some("own answer"))]
#[case::user(owned("owner", &[]), Some("own answer"))]
#[case::team(owned("", &["alpha"]), Some("own answer"))]
#[case::no_identity(owned("", &[]), None)]
#[tokio::test]
async fn agent_final_answer_preserves_visibility_and_trace_ownership(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] access: QueryScope,
    #[case] expected: Option<&str>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let rows = [
        ("alpha", "one", "owner", "root", "", "agent", ""),
        (
            "alpha",
            "one",
            "owner",
            "child",
            "root",
            "llm",
            "own answer",
        ),
        (
            "alpha",
            "two",
            "other",
            "child",
            "root",
            "llm",
            "other key answer",
        ),
        (
            "beta",
            "one",
            "other",
            "child",
            "root",
            "llm",
            "other team answer",
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (team, key, user, span, parent, kind, output))| {
        serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp + index as i64, "TraceId": "shared", "SpanId": span,
            "ParentSpanId": parent, "TeamId": team, "ApiKeyHash": key, "UserId": user,
            "ObservationType": kind, "Input": "prompt", "Output": output,
        }))
    })
    .collect::<Result<Vec<_>, _>>()?;
    insert_rows(&database, "otel_traces", rows).await?;
    let reader = Connection::reader(&database.url, "trace_test")?;
    let detail = TraceReader::new(usize::MAX)
        .get_span(
            &traces(&database, &reader),
            &access,
            "shared",
            "root",
            &trace_ref("alpha", "one", "shared"),
        )
        .await?;
    assert_eq!(
        detail
            .as_ref()
            .map(|detail| (detail.input.as_str(), detail.output.as_str())),
        expected.map(|output| ("prompt", output))
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn nullable_spend_upgrade_preserves_existing_costs_and_unknown_new_costs(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    let timestamp = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64;
    let statements = schema_statements("trace_test", 7)?;
    for statement in &statements[..14] {
        execute_write(&database, statement).await?;
    }
    let legacy = serde_json::from_value(serde_json::json!({
        "request_id": "legacy", "response_id": "legacy-response", "spend": 0.25,
        "start_time": timestamp, "end_time": timestamp + 100
    }))?;
    insert_rows(&database, "spend_logs", vec![legacy]).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let unknown = serde_json::from_value(serde_json::json!({
        "request_id": "unknown", "response_id": "unknown-response", "spend": null,
        "start_time": timestamp, "end_time": timestamp + 100
    }))?;
    let free = serde_json::from_value(serde_json::json!({
        "request_id": "free", "response_id": "free-response", "spend": 0.0,
        "start_time": timestamp, "end_time": timestamp + 100
    }))?;
    insert_rows(&database, "spend_logs", vec![unknown, free]).await?;
    let result = read_json(
        &database,
        "SELECT request_id, spend FROM trace_test.spend_logs FINAL ORDER BY request_id",
    )
    .await?;
    #[derive(Debug, serde::Deserialize)]
    struct CostRow {
        request_id: String,
        spend: Option<f64>,
    }
    let rows: Vec<CostRow> = serde_json::from_value(result["data"].clone())?;
    assert_eq!(
        rows.iter()
            .map(|row| (row.request_id.as_str(), row.spend))
            .collect::<Vec<_>>(),
        vec![
            ("free", Some(0.0)),
            ("legacy", Some(0.25)),
            ("unknown", None)
        ]
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn gateway_id_upgrade_preserves_legacy_rows_and_accepts_new_ids(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    let timestamp = (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64;
    let statements = schema_statements("trace_test", 7)?;
    for statement in &statements[..15] {
        execute_write(&database, statement).await?;
    }
    let legacy = serde_json::from_value(serde_json::json!({
        "request_id": "legacy", "response_id": "response", "spend": 0.25,
        "start_time": timestamp, "end_time": timestamp + 100
    }))?;
    insert_rows(&database, "spend_logs", vec![legacy]).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let current = serde_json::from_value(serde_json::json!({
        "request_id": "current", "response_id": "response", "litellm_call_id": "gateway",
        "spend": null, "start_time": timestamp, "end_time": timestamp + 100
    }))?;
    insert_rows(&database, "spend_logs", vec![current]).await?;
    let result = read_json(&database,
        "SELECT request_id, litellm_call_id, spend FROM trace_test.spend_logs FINAL ORDER BY request_id"
    ).await?;
    assert_eq!(
        result["data"],
        serde_json::json!([
            {"request_id": "current", "litellm_call_id": "gateway", "spend": null},
            {"request_id": "legacy", "litellm_call_id": "", "spend": 0.25},
        ])
    );
    Ok(())
}
