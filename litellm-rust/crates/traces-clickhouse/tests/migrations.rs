use std::{collections::BTreeMap, time::Duration};

use litellm_http::Client;
use litellm_traces_clickhouse::{
    Connection, Error, InsertTable, NORMALIZED_FIELD_DEFINITIONS, Parameter, ReadQuery,
    encode_rows, ensure_schema, execute_named_read, execute_read, schema_statements,
};
use rstest::rstest;
mod support;

use support::{ClickHouseDatabase, TestResult, database};

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
    let detail =
        litellm_storage_clickhouse::fetch::<litellm_traces_clickhouse::query::named::SpanDetail>(
            &database.client,
            &reader,
            &litellm_traces_clickhouse::query::named::SpanDetailParams {
                access: litellm_traces_clickhouse::query::named::ReadAccessParams {
                    all_teams: 0,
                    user_id: String::new(),
                    team_ids: vec!["team-1".into()],
                },
                trace_id: "trace-1".into(),
                trace_ref: String::new(),
                span_id: "span-1".into(),
            },
        )
        .await?;
    assert_eq!(detail.len(), 1);
    assert_eq!(detail[0].input, "hello world");
    assert_eq!(detail[0].attributes["gen_ai.response.id"], "response-1");
    let list_parameters = BTreeMap::from([
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec!["team-1".into()])),
        (
            "start_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 - 1000),
        ),
        (
            "end_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 + 1000),
        ),
        ("cursor_ms".into(), Parameter::Integer(0)),
        ("cursor_trace_id".into(), Parameter::Text(String::new())),
        ("limit".into(), Parameter::Integer(10)),
    ]);
    let listed: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &reader,
            ReadQuery::ListTraces,
            &list_parameters,
        )
        .await?,
    )?;
    assert_eq!(
        listed["data"][0]["request_ids"],
        serde_json::json!(["response-1"])
    );
    let spend_parameters = BTreeMap::from([
        (
            "response_ids".into(),
            Parameter::Strings(vec!["response-1".into()]),
        ),
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec!["team-1".into()])),
        (
            "start_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 - 1000),
        ),
        (
            "end_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 + 1000),
        ),
    ]);
    let matched: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &reader,
            ReadQuery::SpendByResponseIds,
            &spend_parameters,
        )
        .await?,
    )?;
    assert_eq!(matched["data"][0]["spend"], 0.125);
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
async fn retried_trace_insert_does_not_inflate_rollup(
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
    let parameters = BTreeMap::from([
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text("owner".into())),
        ("team_ids".into(), Parameter::Strings(vec![])),
        (
            "start_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 - 1000),
        ),
        (
            "end_ms".into(),
            Parameter::Integer(timestamp / 1_000_000 + 1000),
        ),
        ("cursor_ms".into(), Parameter::Integer(0)),
        ("cursor_trace_id".into(), Parameter::Text(String::new())),
        ("limit".into(), Parameter::Integer(1)),
    ]);
    let first: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::ListTraces,
            &parameters,
        )
        .await?,
    )?;
    let cursor = first["data"][0]["trace_ref"]
        .as_str()
        .ok_or("missing cursor")?;
    let next_parameters = parameters
        .into_iter()
        .chain([
            (
                "cursor_ms".into(),
                Parameter::Integer(timestamp / 1_000_000),
            ),
            ("cursor_trace_id".into(), Parameter::Text(cursor.into())),
        ])
        .collect();
    let second: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::ListTraces,
            &next_parameters,
        )
        .await?,
    )?;
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
        serde_json::json!(["research_agent", "reviewer"])
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
    for page in [&first, &second] {
        assert!(
            page["statistics"]["rows_read"]
                .as_u64()
                .ok_or("missing read statistics")?
                < 5000
        );
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn rollup_merges_spans_across_days_without_losing_root_fields(
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
    insert_rows(&database, "otel_traces", vec![root]).await?;
    let child = serde_json::from_value(serde_json::json!({
        "Timestamp": day_start + 1_000_000_000, "TraceId": "cross-day", "SpanId": "span-child",
        "ParentSpanId": "span-root", "ServiceName": "proxy", "SpanName": "child",
        "AgentName": "researcher", "ObservationType": "agent",
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
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let parameters = BTreeMap::from([
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec!["team-1".into()])),
        (
            "start_ms".into(),
            Parameter::Integer(day_start / 1_000_000 - 2000),
        ),
        ("end_ms".into(), Parameter::Integer(day_start / 1_000_000)),
        ("cursor_ms".into(), Parameter::Integer(0)),
        ("cursor_trace_id".into(), Parameter::Text(String::new())),
        ("limit".into(), Parameter::Integer(10)),
    ]);
    let listed: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::ListTraces,
            &parameters,
        )
        .await?,
    )?;
    assert_eq!(
        listed["data"][0]["agent_names"],
        serde_json::json!(["lead", "researcher"])
    );
    assert_eq!(listed["data"][0]["agent_count"], 2);
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
            {"name": "agent_traces_by_key"},
            {"name": "otel_traces"},
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
    assert_eq!(table_rows(&database, "agent_traces_by_key").await?, 1);
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
async fn lens_filters_reads_and_evidence_keep_reused_trace_ids_separate(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    use litellm_traces_clickhouse::{Parameter, ReadQuery};
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
    let parameters = BTreeMap::from([
        ("source".into(), Parameter::Text("traces".into())),
        ("all_teams".into(), Parameter::Integer(1)),
        ("team".into(), Parameter::Text(String::new())),
        ("key_hash".into(), Parameter::Text(String::new())),
        (
            "start".into(),
            Parameter::Integer(timestamp / 1_000_000 - 1000),
        ),
        (
            "end".into(),
            Parameter::Integer(timestamp / 1_000_000 + 1000),
        ),
        ("agent_name".into(), Parameter::Text(String::new())),
        ("service".into(), Parameter::Text("review".into())),
        (
            "filter_keys".into(),
            Parameter::Strings(vec!["swarm".into()]),
        ),
        (
            "filter_values".into(),
            Parameter::Strings(vec!["release".into()]),
        ),
        ("limit".into(), Parameter::Integer(10)),
        ("offset".into(), Parameter::Integer(0)),
        ("after".into(), Parameter::Text(String::new())),
        ("sample_percent".into(), Parameter::Text("100".into())),
        ("sample_cap".into(), Parameter::Integer(0)),
        ("preview".into(), Parameter::Integer(0)),
        ("selected_team".into(), Parameter::Text(String::new())),
        ("execution_ids".into(), Parameter::Strings(vec![])),
    ]);
    let sample: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Sample,
            &parameters,
        )
        .await?,
    )?;
    let rows = sample["data"].as_array().expect("sample rows");
    assert_eq!(rows.len(), 2);
    assert_ne!(rows[0]["trace_ref"], rows[1]["trace_ref"]);
    let identity_params = BTreeMap::from([
        ("trace_id".into(), Parameter::Text("shared".into())),
        ("all_teams".into(), Parameter::Integer(0)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec!["team".into()])),
    ]);
    let identities: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::TraceIdentity,
            &identity_params,
        )
        .await?,
    )?;
    assert_eq!(identities["data"].as_array().map(Vec::len), Some(2));
    let user_params = identity_params
        .into_iter()
        .chain([
            ("team_ids".into(), Parameter::Strings(vec![])),
            ("user_id".into(), Parameter::Text("one".into())),
        ])
        .collect();
    let identity: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::TraceIdentity,
            &user_params,
        )
        .await?,
    )?;
    assert_eq!(identity["data"].as_array().map(Vec::len), Some(1));
    assert!(
        rows.iter()
            .any(|row| row["trace_ref"] == identity["data"][0]["trace_ref"])
    );
    let first_ref = rows[0]["trace_ref"].as_str().expect("reference");
    let read_parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .chain([
            ("id".into(), Parameter::Text("shared".into())),
            ("record_team".into(), Parameter::Text("team".into())),
            ("trace_ref".into(), Parameter::Text(first_ref.into())),
            ("cursor".into(), Parameter::Text(String::new())),
            ("offset".into(), Parameter::Integer(1)),
            ("span".into(), Parameter::Text("root".into())),
        ])
        .collect();
    let content: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Content,
            &read_parameters,
        )
        .await?,
    )?;
    assert_eq!(content["data"].as_array().map(Vec::len), Some(1));
    let text = content["data"][0]["content"].as_str().expect("content");
    let opposite = if text.contains("timeout") {
        "success"
    } else {
        "timeout"
    };
    let evidence_parameters = read_parameters
        .into_iter()
        .chain([("quote".into(), Parameter::Text(opposite.into()))])
        .collect();
    let evidence: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Evidence,
            &evidence_parameters,
        )
        .await?,
    )?;
    assert_eq!(evidence["data"][0]["count"], 0);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn lens_request_sample_does_not_trust_caller_tags(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    use litellm_traces_clickhouse::{Parameter, ReadQuery};
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64 / 1_000_000;
    for (id, internal) in [("external", false), ("internal", true)] {
        let row = serde_json::from_value(serde_json::json!({
            "request_id": id, "team_id": "team", "start_time": timestamp, "end_time": timestamp,
            "request_tags": ["litellm-engine"],
            "metadata": serde_json::json!({"litellm_lens_internal": internal}).to_string()
        }))?;
        insert_rows(&database, "spend_logs", vec![row]).await?;
    }
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let parameters = BTreeMap::from([
        ("source".into(), Parameter::Text("requests".into())),
        ("all_teams".into(), Parameter::Integer(1)),
        ("team".into(), Parameter::Text(String::new())),
        ("key_hash".into(), Parameter::Text(String::new())),
        ("start".into(), Parameter::Integer(timestamp - 1000)),
        ("end".into(), Parameter::Integer(timestamp + 60000)),
        ("agent_name".into(), Parameter::Text(String::new())),
        ("service".into(), Parameter::Text(String::new())),
        ("filter_keys".into(), Parameter::Strings(vec![])),
        ("filter_values".into(), Parameter::Strings(vec![])),
        ("limit".into(), Parameter::Integer(10)),
        ("offset".into(), Parameter::Integer(0)),
        ("after".into(), Parameter::Text(String::new())),
        ("sample_percent".into(), Parameter::Text("100".into())),
        ("sample_cap".into(), Parameter::Integer(0)),
        ("preview".into(), Parameter::Integer(0)),
        ("selected_team".into(), Parameter::Text(String::new())),
        ("execution_ids".into(), Parameter::Strings(vec![])),
    ]);
    let sample: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Sample,
            &parameters,
        )
        .await?,
    )?;
    let rows = sample["data"].as_array().expect("sample rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["trace_id"], "external");
    Ok(())
}

#[rstest]
#[case::changing("100", 0, 0, 1001, 100, true)]
#[case::all("100", 0, 0, 1001, 100, false)]
#[case::percentage("10", 0, 0, 101, 100, false)]
#[case::capped("100", 25, 0, 25, 100, false)]
#[case::preview("10", 25, 1, 1001, 100, false)]
#[tokio::test]
async fn lens_selection_pages_without_losing_or_repeating_runs(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] percent: &str,
    #[case] cap: i64,
    #[case] preview: i64,
    #[case] expected: usize,
    #[case] page_size: usize,
    #[case] changing: bool,
) -> TestResult {
    use litellm_traces_clickhouse::ReadQuery;
    let database = database?;
    ensure_schema(
        &database.client,
        &Connection::writer(&database.url)?,
        "trace_test",
        7,
    )
    .await?;
    execute_write(&database, "INSERT INTO trace_test.spend_logs (request_id,team_id,start_time,end_time) SELECT toString(number),'team',now64(3)-INTERVAL 5 MINUTE,now64(3)-INTERVAL 5 MINUTE FROM numbers(1001)").await?;
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let end = time::OffsetDateTime::now_utc().unix_timestamp() * 1000 + 60000;
    let mut seen = std::collections::BTreeSet::new();
    let mut cursor = String::new();
    let step = if page_size == 0 { expected } else { page_size };
    for offset in (0..expected).step_by(step) {
        let parameters = BTreeMap::from([
            ("source".into(), Parameter::Text("requests".into())),
            ("all_teams".into(), Parameter::Integer(0)),
            ("team".into(), Parameter::Text("team".into())),
            ("key_hash".into(), Parameter::Text(String::new())),
            ("start".into(), Parameter::Integer(0)),
            ("end".into(), Parameter::Integer(end)),
            ("agent_name".into(), Parameter::Text(String::new())),
            ("service".into(), Parameter::Text(String::new())),
            ("filter_keys".into(), Parameter::Strings(vec![])),
            ("filter_values".into(), Parameter::Strings(vec![])),
            ("limit".into(), Parameter::Integer(page_size as i64)),
            (
                "offset".into(),
                Parameter::Integer(if changing { 0 } else { offset as i64 }),
            ),
            ("after".into(), Parameter::Text(cursor.clone())),
            ("sample_percent".into(), Parameter::Text(percent.into())),
            ("sample_cap".into(), Parameter::Integer(cap)),
            ("preview".into(), Parameter::Integer(preview)),
            ("selected_team".into(), Parameter::Text(String::new())),
            ("execution_ids".into(), Parameter::Strings(vec![])),
        ]);
        let body = execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Sample,
            &parameters,
        )
        .await?;
        let json: serde_json::Value = serde_json::from_str(&body)?;
        let rows = json["data"].as_array().expect("sample rows");
        assert_eq!(rows.len(), step.min(expected - offset));
        for row in rows {
            assert_eq!(
                row["eligible"],
                if changing && offset > 0 { 1000 } else { 1001 }
            );
            assert!(seen.insert(row["trace_id"].as_str().expect("run id").to_owned()));
        }
        if changing {
            cursor = rows.last().expect("last run")["selection_key"]
                .as_str()
                .expect("selection key")
                .to_owned();
            if offset == 0 {
                let removed = rows[0]["trace_id"].as_str().expect("request id");
                execute_write(&database, &format!("ALTER TABLE trace_test.spend_logs DELETE WHERE request_id='{removed}' SETTINGS mutations_sync=1")).await?;
            }
        }
    }
    assert_eq!(seen.len(), expected);
    Ok(())
}

#[rstest]
#[case::short(100)]
#[case::boundary(7970)]
#[case::long(16000)]
#[tokio::test]
async fn lens_content_keeps_output_visible_after_long_input(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] input_length: usize,
) -> TestResult {
    use litellm_traces_clickhouse::ReadQuery;
    let database = database?;
    ensure_schema(
        &database.client,
        &Connection::writer(&database.url)?,
        "trace_test",
        7,
    )
    .await?;
    insert_rows(&database, "spend_logs", vec![serde_json::from_value(serde_json::json!({
        "request_id": "request", "team_id": "team", "start_time": time::OffsetDateTime::now_utc().unix_timestamp()*1000, "end_time": time::OffsetDateTime::now_utc().unix_timestamp()*1000, "messages": "x".repeat(input_length), "response": "Delivered result"
    }))?]).await?;
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let mut parameters = BTreeMap::from([
        ("source".into(), Parameter::Text("requests".into())),
        ("all_teams".into(), Parameter::Integer(0)),
        ("team".into(), Parameter::Text("team".into())),
        ("record_team".into(), Parameter::Text("team".into())),
        ("key_hash".into(), Parameter::Text(String::new())),
        ("trace_ref".into(), Parameter::Text(String::new())),
        ("id".into(), Parameter::Text("request".into())),
        ("cursor".into(), Parameter::Text(String::new())),
        ("offset".into(), Parameter::Integer(1)),
    ]);
    let body = execute_named_read(
        &database.client,
        &connection,
        ReadQuery::Content,
        &parameters,
    )
    .await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let text = json["data"][0]["content"].as_str().expect("content");
    assert!(text.contains("Output: Delivered result"));
    assert!(text.len() <= 8000);
    assert_eq!(
        json["data"][0]["truncated"],
        u8::from(input_length + "Input: \nOutput: Delivered result\nError: ".len() > 8000)
    );
    let original = format!(
        "Input: {}\nOutput: Delivered result\nError: ",
        "x".repeat(input_length)
    );
    let mut recovered = String::new();
    for offset in (2..original.len() + 2).step_by(8000) {
        parameters.insert("offset".into(), Parameter::Integer(offset as i64));
        let body = execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Content,
            &parameters,
        )
        .await?;
        let page: serde_json::Value = serde_json::from_str(&body)?;
        recovered.push_str(page["data"][0]["content"].as_str().expect("content"));
    }
    assert_eq!(recovered, original);
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
    let mut parameters = BTreeMap::from([
        (
            "trace_id".into(),
            Parameter::Text("diagnostic-trace".into()),
        ),
        ("all_teams".into(), Parameter::Integer(1)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec![])),
        ("trace_ref".into(), Parameter::Text(String::new())),
    ]);
    let body = execute_named_read(
        &database.client,
        &reader,
        ReadQuery::TraceSpans,
        &parameters,
    )
    .await?;
    let response: serde_json::Value = serde_json::from_str(&body)?;
    let spans = response["data"].as_array().expect("trace spans");
    assert_eq!(spans.len(), span_count);
    let prefix: String = message.chars().take(128).collect();
    assert!(!prefix.is_empty());
    assert!(
        spans
            .iter()
            .all(|span| span["status_message"] == prefix && span["error_truncated"] == 1)
    );
    parameters.insert("span_id".into(), Parameter::Text("span-0".into()));
    parameters.insert("error_version".into(), Parameter::Text(String::new()));
    let mut recovered = String::new();
    loop {
        parameters.insert(
            "error_offset".into(),
            Parameter::Integer(recovered.chars().count() as i64),
        );
        let body = execute_named_read(&database.client, &reader, ReadQuery::SpanError, &parameters)
            .await?;
        assert!(body.len() < 128 * 1024);
        let response: serde_json::Value = serde_json::from_str(&body)?;
        let chunk = response["data"][0]["message"]
            .as_str()
            .expect("diagnostic chunk");
        assert!(!chunk.is_empty());
        recovered.push_str(chunk);
        let version = response["data"][0]["version"]
            .as_str()
            .expect("diagnostic version");
        parameters.insert("error_version".into(), Parameter::Text(version.into()));
        if recovered.chars().count() >= message.chars().count() {
            break;
        }
    }
    assert_eq!(recovered, message);
    parameters.insert("all_teams".into(), Parameter::Integer(0));
    parameters.insert("user_id".into(), Parameter::Text("unrelated-user".into()));
    let denied =
        execute_named_read(&database.client, &reader, ReadQuery::SpanError, &parameters).await?;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&denied)?["data"],
        serde_json::json!([])
    );
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
    let parameters = BTreeMap::from([
        ("trace_id".into(), Parameter::Text("duplicate-trace".into())),
        ("span_id".into(), Parameter::Text("duplicate-span".into())),
        ("all_teams".into(), Parameter::Integer(1)),
        ("user_id".into(), Parameter::Text(String::new())),
        ("team_ids".into(), Parameter::Strings(vec![])),
        ("trace_ref".into(), Parameter::Text(String::new())),
        ("error_version".into(), Parameter::Text(String::new())),
        ("error_offset".into(), Parameter::Integer(0)),
    ]);
    let preview = execute_named_read(
        &database.client,
        &reader,
        ReadQuery::TraceSpans,
        &parameters,
    )
    .await?;
    let diagnostic =
        execute_named_read(&database.client, &reader, ReadQuery::SpanError, &parameters).await?;
    let preview: serde_json::Value = serde_json::from_str(&preview)?;
    let diagnostic: serde_json::Value = serde_json::from_str(&diagnostic)?;
    assert_eq!(preview["data"].as_array().unwrap().len(), 1);
    assert_eq!(preview["data"][0]["status_message"], message[..128]);
    assert_eq!(diagnostic["data"][0]["message"], message);
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
#[tokio::test]
async fn lens_agent_discovery_and_selection_preserve_scope(
    #[future] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    use litellm_traces_clickhouse::ReadQuery;
    let database = database.await?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    for (team, key, trace, agent, span, parent) in [
        ("alpha", "one", "research", "research_agent", "root", ""),
        ("alpha", "one", "research", "", "tool", "root"),
        ("alpha", "one", "support", "support_agent", "root", ""),
        ("alpha", "two", "hidden-key", "private_agent", "root", ""),
        ("beta", "one", "hidden-team", "other_agent", "root", ""),
    ] {
        insert_rows(
            &database,
            "otel_traces",
            vec![serde_json::from_value(serde_json::json!({
                "Timestamp": timestamp, "TraceId": trace, "SpanId": span, "ParentSpanId": parent,
                "ServiceName": "shared-app", "SpanName": "run", "Input": "test",
                "SpanAttributes": {"gen_ai.agent.name": agent},
                "ResourceAttributes": {"litellm.team_id": team, "litellm.api_key_hash": key}
            }))?],
        )
        .await?;
    }
    let connection = Connection::configured(&database.url, "trace_test", "default", "")?;
    let scope_parameters = BTreeMap::from([
        ("all_teams".into(), Parameter::Integer(0)),
        ("team".into(), Parameter::Text("alpha".into())),
        ("key_hash".into(), Parameter::Text("one".into())),
    ]);
    let agents: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Agents,
            &scope_parameters,
        )
        .await?,
    )?;
    assert_eq!(
        agents["data"],
        serde_json::json!([
            {"agent_name": "research_agent"}, {"agent_name": "support_agent"}
        ])
    );
    let parameters = scope_parameters
        .into_iter()
        .chain([
            ("source".into(), Parameter::Text("traces".into())),
            (
                "start".into(),
                Parameter::Integer(timestamp / 1_000_000 - 1000),
            ),
            (
                "end".into(),
                Parameter::Integer(timestamp / 1_000_000 + 1000),
            ),
            ("service".into(), Parameter::Text("shared-app".into())),
            (
                "agent_name".into(),
                Parameter::Text("research_agent".into()),
            ),
            ("filter_keys".into(), Parameter::Strings(vec![])),
            ("filter_values".into(), Parameter::Strings(vec![])),
            ("limit".into(), Parameter::Integer(100)),
            ("offset".into(), Parameter::Integer(0)),
            ("after".into(), Parameter::Text(String::new())),
            ("sample_percent".into(), Parameter::Text("100".into())),
            ("sample_cap".into(), Parameter::Integer(0)),
            ("preview".into(), Parameter::Integer(1)),
            ("selected_team".into(), Parameter::Text(String::new())),
            ("execution_ids".into(), Parameter::Strings(vec![])),
        ])
        .collect::<BTreeMap<_, _>>();
    let sample: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Sample,
            &parameters,
        )
        .await?,
    )?;
    assert_eq!(sample["data"].as_array().expect("rows").len(), 1);
    assert_eq!(sample["data"][0]["trace_id"], "research");
    assert_eq!(sample["data"][0]["span_count"], 2);
    let available: serde_json::Value = serde_json::from_str(
        &execute_named_read(
            &database.client,
            &connection,
            ReadQuery::Availability,
            &parameters,
        )
        .await?,
    )?;
    assert_eq!(available["data"][0]["traces"], 1);
    assert_eq!(available["data"][0]["requests"], 0);
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
    for table in ["otel_traces", "agent_traces_by_key", "spend_logs"] {
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
            "dotted.key": "literal", "quote'\\key": null, "items": [{"name": "first"}],
            "<custom>&{{key}}": {"nested.key": true}
        });
        insert_rows(
            &database,
            "spend_logs",
            vec![serde_json::from_value(serde_json::json!({
                "request_id": "request-1", "response_id": "response-1", "team_id": "team-1",
                "api_key": "key-1", "metadata": metadata.to_string(), "spend": 0.25,
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
    for table in ["otel_traces", "agent_traces_by_key", "spend_logs"] {
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
    for gotcha in help["gotchas"].as_array().ok_or("missing gotchas")? {
        assert!(guide.contains(gotcha.as_str().ok_or("gotcha text")?));
    }
    let tables = help["tables"].as_array().ok_or("missing tables")?;
    assert_eq!(tables.len(), 3);
    let columns = tables[0]["columns"].as_array().ok_or("missing columns")?;
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
        assert!(guide.contains("JSONExtractRaw(metadata, '<custom>&{{key}}', 'nested.key')"));
        assert!(guide.contains("SpanAttributes['custom.tag']"));
        assert!(guide.contains("ResourceAttributes['custom.resource']"));
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
    for example in help["examples"].as_array().ok_or("missing examples")? {
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
            !populated,
            "{sql}"
        );
        if populated && example["name"] == "Traces correlated with LLM call metadata" {
            assert_eq!(values["data"][0]["TraceId"], "trace-1");
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
    for table in ["otel_traces", "agent_traces_by_key", "spend_logs"] {
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
    assert_eq!(help["tables"].as_array().ok_or("tables")?.len(), 3);
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
fn field_definitions_match_serialized_normalized_span() {
    use litellm_traces::decode_otlp;
    use std::collections::BTreeSet;
    let spans = decode_otlp(
        br#"{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"11111111111111111111111111111111","spanId":"2222222222222222","name":"root"}]}]}]}"#,
        Some("application/json"),
    )
    .expect("valid OTLP");
    let fields = &spans[0].normalized;
    let serialized = serde_json::to_value(fields).expect("serializable fields");
    let keys: BTreeSet<_> = serialized
        .as_object()
        .expect("field object")
        .keys()
        .map(String::as_str)
        .collect();
    let mapped: BTreeSet<_> = NORMALIZED_FIELD_DEFINITIONS
        .iter()
        .map(|field| field.name)
        .collect();
    assert_eq!(keys, mapped);
}

#[rstest]
#[case::own_user("owner", vec![], None, vec!["own"])]
#[case::own_user_and_permitted_team("owner", vec!["permitted"], None, vec!["own", "team"])]
#[case::no_identity("", vec![], None, vec![])]
#[case::legacy_key_without_identity("", vec![], Some("request-key"), vec![])]
#[tokio::test]
async fn named_and_sql_readers_share_request_log_visibility(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] user: &str,
    #[case] teams: Vec<&str>,
    #[case] legacy_key: Option<&str>,
    #[case] expected: Vec<&str>,
) -> TestResult {
    use litellm_traces_clickhouse::query::named::{
        ReadAccessParams, SpendByResponseIds, SpendByResponseIdsParams,
    };
    use litellm_traces_clickhouse::{QueryReaders, QueryScope};

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
    let params =
        SpendByResponseIdsParams::from(litellm_traces::query::named::SpendByResponseIdsParams {
            access: serde_json::from_value::<ReadAccessParams>(serde_json::json!({
                "all_teams": 0, "user_id": user, "team_ids": teams,
                "api_key_hash": legacy_key.unwrap_or_default(),
            }))?,
            response_ids: vec!["shared-response".into()],
            start_ms: timestamp / 1_000_000 - 1,
            end_ms: timestamp / 1_000_000 + 1,
        });
    let spend =
        litellm_storage_clickhouse::fetch::<SpendByResponseIds>(&database.client, &reader, &params)
            .await?;
    let actual: std::collections::BTreeSet<_> =
        spend.iter().map(|row| row.0.request_id.as_str()).collect();
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
async fn rollup_cost_completeness_preserves_missing_ids_and_fails_closed_for_historical_rows(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let initial_mutations = mutation_rows(&database).await?;
    let timestamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64;
    let rows = [("complete", "llm", "response"), ("complete", "llm", "response"), ("complete", "agent", ""), ("missing", "llm", "response"), ("missing", "llm", ""), ("missing", "agent", "extra-id"), ("mixed", "llm", "mine"), ("mixed", "llm", "other")]
        .into_iter().enumerate().map(|(index, (trace, kind, id))| serde_json::from_value(serde_json::json!({
            "Timestamp": timestamp, "TraceId": trace, "SpanId": index.to_string(), "TeamId": "team", "ApiKeyHash": "export",
            "UserId": if id == "other" { "other" } else { "owner" }, "ObservationType": kind, "LiteLLMRequestId": id,
        }))).collect::<Result<Vec<BTreeMap<String, serde_json::Value>>, _>>()?;
    insert_rows(&database, "otel_traces", rows).await?;
    execute_write(&database, &format!(
        "INSERT INTO trace_test.agent_traces_by_key (TeamId, ApiKeyHash, TraceId, StartTs, EndTs, LlmCount, RequestIds) \
         VALUES ('team', 'export', 'historical', fromUnixTimestamp64Nano({timestamp}), fromUnixTimestamp64Nano({timestamp}), 2, ['response', 'non-llm-id'])"
    )).await?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    let params = litellm_traces_clickhouse::query::named::ListTracesParams::from(
        litellm_traces::query::named::ListTracesParams {
            access: litellm_traces::query::named::ReadAccessParams {
                all_teams: 0,
                user_id: "".into(),
                team_ids: vec!["team".into()],
            },
            start_ms: timestamp / 1_000_000 - 1,
            end_ms: timestamp / 1_000_000 + 1,
            cursor_ms: 0,
            cursor_trace_id: "".into(),
            limit: 10,
        },
    );
    let reader = Connection::reader(&database.url, "trace_test")?;
    let listed = litellm_storage_clickhouse::fetch::<
        litellm_traces_clickhouse::query::named::ListTraces,
    >(&database.client, &reader, &params)
    .await?;
    assert_eq!(listed.len(), 4);
    let owned_params = litellm_traces_clickhouse::query::named::ListTracesParams::from(
        litellm_traces::query::named::ListTracesParams {
            access: litellm_traces::query::named::ReadAccessParams {
                user_id: "owner".into(),
                team_ids: vec![],
                all_teams: 0,
            },
            ..params.0
        },
    );
    let owned = litellm_storage_clickhouse::fetch::<
        litellm_traces_clickhouse::query::named::ListTraces,
    >(&database.client, &reader, &owned_params)
    .await?;
    assert_eq!(owned.len(), 2);
    assert!(
        owned
            .iter()
            .all(|row| ["complete", "missing"].contains(&row.0.trace_id.as_str()))
    );
    for row in listed {
        match row.0.trace_id.as_str() {
            "complete" => {
                assert_eq!(row.0.user_id, "owner");
                assert_eq!(row.0.request_ids, ["response"]);
                assert_eq!(row.0.llm_calls, 2);
            }
            "missing" | "historical" => assert!(row.0.request_ids.iter().any(String::is_empty)),
            "mixed" => assert!(row.0.user_id.is_empty()),
            id => panic!("unexpected trace {id}"),
        }
    }
    assert_eq!(mutation_rows(&database).await?, initial_mutations);
    Ok(())
}

#[rstest]
#[case::admin(1, "", vec![], "own answer")]
#[case::user(0, "owner", vec![], "own answer")]
#[case::team(0, "", vec!["alpha"], "own answer")]
#[case::no_identity(0, "", vec![], "")]
#[tokio::test]
async fn agent_final_answer_preserves_visibility_and_trace_ownership(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
    #[case] all_teams: u8,
    #[case] user: &str,
    #[case] teams: Vec<&str>,
    #[case] expected: &str,
) -> TestResult {
    use litellm_traces_clickhouse::query::named::{ReadAccessParams, SpanDetail, SpanDetailParams};

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
    let details = litellm_storage_clickhouse::fetch::<SpanDetail>(
        &database.client,
        &reader,
        &SpanDetailParams {
            access: ReadAccessParams {
                all_teams,
                user_id: user.into(),
                team_ids: teams.into_iter().map(str::to_owned).collect(),
            },
            trace_id: "shared".into(),
            trace_ref: String::new(),
            span_id: "root".into(),
        },
    )
    .await?;
    if expected.is_empty() {
        assert!(details.is_empty());
    } else {
        assert_eq!(details.len(), 1);
        assert_eq!(details[0].input, "prompt");
        assert_eq!(details[0].output, expected);
    }
    Ok(())
}
