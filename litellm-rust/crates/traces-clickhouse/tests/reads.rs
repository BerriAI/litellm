use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_storage_clickhouse::fetch;
use litellm_traces::query::named::{self as contracts, ReadAccessParams};
use litellm_traces_cache::{ReadError, TraceListQuery, TraceReader};
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, InsertTable, Parameter, QueryScope, ReadQuery,
    execute_named_read, insert_rows,
    query::named::{ListTraces, ListTracesParams},
};
use rstest::rstest;
use serde_json::json;

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database, seeded_database};
use support::TestResult;

fn make_reader(client: &Client, connection: Connection) -> (TraceReader, ClickHouseTraces) {
    (
        TraceReader::new(litellm_storage_clickhouse::READ_LIMITS.response_bytes),
        ClickHouseTraces::new(client.clone(), connection),
    )
}

#[rstest]
#[case::api_key("key-a", "")]
#[case::user("", "user-a")]
#[tokio::test]
async fn list_costs_match_each_run_when_response_ids_are_reused(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] api_key: &str,
    #[case] user_id: &str,
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let runs = [
        ("earlier-run", 1_790_000_000_000_i64, 0.25),
        ("later-run", 1_790_007_200_000_i64, 0.75),
    ];
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        runs.iter()
            .map(|(trace_id, start_ms, _)| {
                BTreeMap::from([
                    ("Timestamp".into(), json!(start_ms * 1_000_000)),
                    ("TraceId".into(), json!(trace_id)),
                    ("SpanId".into(), json!("llm-span")),
                    ("ObservationType".into(), json!("llm")),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!(api_key)),
                    ("UserId".into(), json!(user_id)),
                    ("Duration".into(), json!(1_000_000)),
                    ("LiteLLMRequestId".into(), json!("reused-response")),
                    ("CallEvidence".into(), json!("complete")),
                ])
            })
            .collect(),
    )
    .await?;
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::SpendLogs,
        runs.iter()
            .map(|(trace_id, start_ms, cost)| {
                BTreeMap::from([
                    ("request_id".into(), json!(format!("request-{trace_id}"))),
                    ("response_id".into(), json!("reused-response")),
                    ("team_id".into(), json!("team-a")),
                    ("api_key".into(), json!(api_key)),
                    ("user".into(), json!(user_id)),
                    ("start_time".into(), json!(start_ms)),
                    ("end_time".into(), json!(start_ms + 1)),
                    ("spend".into(), json!(cost)),
                ])
            })
            .collect(),
    )
    .await?;
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: user_id.into(),
        team_ids: vec!["team-a".into()],
    };
    let page = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    assert_eq!(page.data.len(), runs.len());
    for (trace_id, _, cost) in runs {
        let summary = page
            .data
            .iter()
            .find(|summary| summary.trace_id == trace_id)
            .ok_or("missing run")?;
        let detail = reader
            .get_trace(&store, &access, trace_id, &summary.trace_ref)
            .await?
            .ok_or("missing trace")?;
        assert_eq!(detail.summary.spend, Some(cost));
        assert_eq!(summary.spend, detail.summary.spend, "{trace_id}");
    }
    Ok(())
}

#[rstest]
#[case::many_runs(50, 21, 0, false)]
#[case::one_large_run(1, 1100, 0, false)]
#[case::large_rows(1, 280, 20_000, false)]
#[case::large_cached_snapshot(1, 280, 140_000, false)]
#[case::many_costs(1, 1101, 0, true)]
#[case::many_costed_runs(500, 2, 0, true)]
#[tokio::test]
async fn large_runs_remain_complete_under_default_reader_limits(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] runs: usize,
    #[case] steps: usize,
    #[case] name_bytes: usize,
    #[case] costed: bool,
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let rows = (0..runs)
        .flat_map(|run| {
            (0..steps).map(move |step| {
                BTreeMap::from([
                    (
                        "Timestamp".into(),
                        json!(1_790_000_000_000_000_000_i64 + step as i64),
                    ),
                    ("TraceId".into(), json!(format!("trace-{run:04}"))),
                    ("SpanId".into(), json!(format!("span-{step:04}"))),
                    (
                        "ParentSpanId".into(),
                        json!(if step == 0 { "" } else { "span-0000" }),
                    ),
                    (
                        "SpanName".into(),
                        json!(if name_bytes == 0 {
                            format!("step-{step}")
                        } else {
                            "x".repeat(name_bytes)
                        }),
                    ),
                    (
                        "ObservationType".into(),
                        json!(if step == 0 {
                            "agent"
                        } else if costed {
                            "llm"
                        } else {
                            "tool"
                        }),
                    ),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                    ("Duration".into(), json!(1000)),
                    (
                        "CallEvidence".into(),
                        json!(if costed && step > 0 {
                            "complete"
                        } else {
                            "unknown"
                        }),
                    ),
                    (
                        "LiteLLMRequestId".into(),
                        json!(if costed && step > 0 {
                            format!("response-{step}")
                        } else {
                            String::new()
                        }),
                    ),
                ])
            })
        })
        .collect::<Vec<_>>();
    for chunk in rows.chunks(100) {
        insert_rows(
            client,
            &writer,
            DATABASE,
            InsertTable::OtelTraces,
            chunk.to_vec(),
        )
        .await?;
    }
    if costed {
        let costs = (1..steps)
            .map(|step| {
                BTreeMap::from([
                    ("request_id".into(), json!(format!("request-{step}"))),
                    ("response_id".into(), json!(format!("response-{step}"))),
                    ("team_id".into(), json!("team-a")),
                    ("api_key".into(), json!("key-a")),
                    ("start_time".into(), json!(1_790_000_000_000_i64)),
                    ("end_time".into(), json!(1_790_000_000_001_i64)),
                    ("spend".into(), json!(0.25)),
                ])
            })
            .collect::<Vec<_>>();
        insert_rows(client, &writer, DATABASE, InsertTable::SpendLogs, costs).await?;
    }
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into()],
    };
    let page = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 500,
                agent: "",
            },
        )
        .await?;
    assert_eq!(page.data.len(), runs);
    assert!(
        page.data
            .windows(2)
            .all(|runs| runs[0].trace_ref > runs[1].trace_ref)
    );
    if runs > 1 {
        client
            .post(writer.url().clone())
            .body("SYSTEM FLUSH LOGS")
            .send()
            .await?
            .error_for_status()?;
        for table in ["otel_traces AS o", "spend_logs FINAL"]
            .into_iter()
            .take(if costed { 2 } else { 1 })
        {
            let read_queries = client.post(writer.url().clone()).body(format!(
                "SELECT count() FROM system.query_log WHERE type = 'QueryFinish' AND current_database = '{DATABASE}' AND query LIKE '%FROM {table}%' AND query NOT LIKE '%system.query_log%'"
            )).send().await?.error_for_status()?.text().await?;
            let read_queries = read_queries.trim().parse::<usize>()?;
            assert!(
                read_queries > 0 && read_queries < runs,
                "{read_queries} {table} queries for {runs} runs"
            );
        }
    }
    for summary in &page.data {
        assert_eq!(summary.span_count, steps as u64);
        assert_eq!(
            if costed {
                summary.llm_calls
            } else {
                summary.tool_calls
            },
            (steps - 1) as u64
        );
        if costed {
            assert_eq!(summary.spend, Some((steps - 1) as f64 * 0.25));
        }
    }
    let trace_ref = &page
        .data
        .iter()
        .find(|run| run.trace_id == "trace-0000")
        .ok_or("missing run")?
        .trace_ref;
    let detail = reader
        .get_trace(&store, &access, "trace-0000", trace_ref)
        .await?
        .ok_or("missing trace")?;
    assert_eq!(detail.spans.len(), steps);
    assert_eq!(detail.spans[0].span_id, "span-0000");
    assert_eq!(
        detail.spans[steps - 1].span_id,
        format!("span-{:04}", steps - 1)
    );
    assert_eq!(
        if costed {
            detail.summary.llm_calls
        } else {
            detail.summary.tool_calls
        },
        (steps - 1) as u64
    );
    let denied = ReadAccessParams {
        team_ids: vec!["other-team".into()],
        ..access.clone()
    };
    assert!(
        reader
            .get_trace(&store, &denied, "trace-0000", trace_ref)
            .await?
            .is_none()
    );
    let mut cursor = None;
    let mut ids = Vec::new();
    loop {
        let page = reader
            .get_trace_page(
                &store,
                &access,
                "trace-0000",
                trace_ref,
                cursor.as_deref(),
                200,
            )
            .await?
            .ok_or("missing page")?;
        assert_eq!(page.summary, detail.summary);
        assert!(page.spans.len() <= 200);
        assert!(
            serde_json::to_vec(&page)?.len()
                <= litellm_storage_clickhouse::READ_LIMITS.response_bytes
        );
        if ids.is_empty() {
            assert!(
                reader
                    .get_trace_page(
                        &store,
                        &denied,
                        "trace-0000",
                        trace_ref,
                        page.next_cursor.as_deref(),
                        200,
                    )
                    .await?
                    .is_none()
            );
            client
                .post(writer.url().clone())
                .body(format!("TRUNCATE TABLE {DATABASE}.otel_traces"))
                .send()
                .await?
                .error_for_status()?;
        }
        ids.extend(page.spans.into_iter().map(|span| span.span_id));
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(
        ids,
        detail
            .spans
            .iter()
            .map(|span| span.span_id.clone())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn cursor_pages_keep_a_tenant_scoped_snapshot_when_more_spans_arrive(
    #[future(awt)] seeded_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = seeded_database?;
    let client = &fixture.database.client;
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection.clone());
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let listed = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 10,
                agent: "",
            },
        )
        .await?;
    let summary = listed
        .data
        .iter()
        .find(|summary| summary.span_count == 3)
        .ok_or("missing fixture")?;
    let first = reader
        .get_trace_page(
            &store,
            &access,
            &summary.trace_id,
            &summary.trace_ref,
            None,
            1,
        )
        .await?
        .ok_or("missing first page")?;
    let original_ids = reader
        .get_trace(&store, &access, &summary.trace_id, &summary.trace_ref)
        .await?
        .ok_or("missing trace")?
        .spans
        .into_iter()
        .map(|span| span.span_id)
        .collect::<Vec<_>>();
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!(1_790_000_000_000_000_000_i64)),
            ("TraceId".into(), json!(summary.trace_id)),
            ("SpanId".into(), json!("late-span")),
            ("ParentSpanId".into(), json!(first.spans[0].span_id)),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("EngineReceivedMs".into(), json!(u64::MAX / 2)),
        ])],
    )
    .await?;
    let denied = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["not-this-team".into()],
    };
    assert!(
        reader
            .get_trace_page(
                &store,
                &denied,
                &summary.trace_id,
                &summary.trace_ref,
                first.next_cursor.as_deref(),
                1
            )
            .await?
            .is_none()
    );
    let first_cursor = first.next_cursor.clone();
    let mut cursor = first.next_cursor;
    let mut ids = first
        .spans
        .into_iter()
        .map(|span| span.span_id)
        .collect::<Vec<_>>();
    while let Some(current) = cursor {
        let next = reader
            .get_trace_page(
                &store,
                &access,
                &summary.trace_id,
                &summary.trace_ref,
                Some(&current),
                1,
            )
            .await?
            .ok_or("missing next page")?;
        assert_eq!(next.summary.span_count, 3);
        ids.extend(next.spans.into_iter().map(|span| span.span_id));
        cursor = next.next_cursor;
    }
    assert_eq!(ids, original_ids);
    let cached = reader
        .get_trace(&store, &access, &summary.trace_id, &summary.trace_ref)
        .await?
        .ok_or("missing cached trace")?;
    assert_eq!(cached.spans.len(), 3);
    let (fresh_reader, fresh_store) = make_reader(client, connection);
    let refreshed = fresh_reader
        .get_trace(&fresh_store, &access, &summary.trace_id, &summary.trace_ref)
        .await?
        .ok_or("missing refreshed trace")?;
    assert_eq!(refreshed.spans.len(), 4);
    assert!(matches!(
        reader
            .get_trace_page(
                &store,
                &access,
                &summary.trace_id,
                &summary.trace_ref,
                Some("invalid"),
                1
            )
            .await,
        Err(ReadError::InvalidCursor("span"))
    ));
    let backdated = json!({
        "Timestamp": "2026-09-01 00:00:00.000000000",
        "TraceId": summary.trace_id,
        "SpanId": "backdated-span",
        "EngineReceivedMs": 1,
        "TeamId": "team-a",
        "ApiKeyHash": "key-a"
    });
    client
        .post(writer.url().clone())
        .body(format!(
            "INSERT INTO {DATABASE}.otel_traces FORMAT JSONEachRow\n{backdated}"
        ))
        .send()
        .await?
        .error_for_status()?;
    let uncached_connection =
        Connection::reader(&format!("{}?max_threads=1", fixture.database.url), DATABASE)?;
    let (uncached_reader, uncached_store) = make_reader(client, uncached_connection);
    let changed = uncached_reader
        .get_trace_page(
            &uncached_store,
            &access,
            &summary.trace_id,
            &summary.trace_ref,
            first_cursor.as_deref(),
            1,
        )
        .await;
    assert!(
        matches!(changed, Err(ReadError::TraceChanged)),
        "{changed:?}"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn an_oversized_span_keeps_the_run_list_available_with_partial_totals(
    #[future(awt)] seeded_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = seeded_database?;
    let client = &fixture.database.client;
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection.clone());
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let before = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    let run = before
        .data
        .iter()
        .find(|run| run.span_count == 3)
        .ok_or("missing fixture")?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!(1_790_000_000_000_000_000_i64)),
            ("TraceId".into(), json!(run.trace_id)),
            ("SpanId".into(), json!("oversized-child")),
            ("ParentSpanId".into(), json!("0101010101010101")),
            (
                "SpanName".into(),
                json!("x".repeat(litellm_storage_clickhouse::READ_LIMITS.response_bytes + 1)),
            ),
            ("ObservationType".into(), json!("tool")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
        ])],
    )
    .await?;
    let cached = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    assert_eq!(cached.data, before.data);
    let (reader, store) = make_reader(client, connection);
    let after = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    assert_eq!(after.data.len(), before.data.len());
    let limited = after
        .data
        .iter()
        .find(|item| item.trace_ref == run.trace_ref)
        .ok_or("missing run")?;
    assert!(limited.resolution_limited);
    assert_eq!(limited.span_count, 4);
    assert!(
        after
            .data
            .iter()
            .filter(|item| item.trace_ref != run.trace_ref)
            .all(|item| !item.resolution_limited)
    );
    assert!(matches!(
        reader
            .get_trace_page(&store, &access, &run.trace_id, &run.trace_ref, None, 200)
            .await,
        Err(ReadError::TooLarge)
    ));
    Ok(())
}

#[rstest]
#[case::same_key("same-key", Some(0.25))]
#[case::other_key("other-key", None)]
#[case::foreign_team("foreign-team", None)]
#[case::call_id_other_key("call-id-other-key", None)]
#[case::call_id_foreign_team("call-id-foreign-team", None)]
#[case::transport_only("transport-only", None)]
#[tokio::test]
async fn assigned_call_ids_require_shared_ownership_through_detail_and_batch_reads(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] id: &str,
    #[case] expected: Option<f64>,
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let start_ms = 1_790_000_000_000_i64;
    let cases = [
        (
            "same-key",
            "provider_response:same-key",
            "team-a",
            "key-a",
            Some(0.25),
        ),
        (
            "other-key",
            "provider_response:other-key",
            "team-a",
            "key-b",
            None,
        ),
        (
            "foreign-team",
            "provider_response:foreign-team",
            "team-b",
            "key-a",
            None,
        ),
        (
            "call-id-other-key",
            "litellm_request:call-id-other-key",
            "team-a",
            "key-b",
            None,
        ),
        (
            "call-id-foreign-team",
            "litellm_request:call-id-foreign-team",
            "team-b",
            "key-a",
            None,
        ),
        ("transport-only", "transport:", "team-a", "key-a", None),
    ];
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        cases
            .iter()
            .map(|(id, key, _, _, _)| {
                BTreeMap::from([
                    ("Timestamp".into(), json!(start_ms * 1_000_000)),
                    ("Duration".into(), json!(1_000_000)),
                    ("TraceId".into(), json!(id)),
                    ("SpanId".into(), json!("call")),
                    ("ObservationType".into(), json!("llm")),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                    ("CallKeys".into(), json!([key])),
                    ("CallEvidence".into(), json!("complete")),
                ])
            })
            .collect(),
    )
    .await?;
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::SpendLogs,
        cases
            .iter()
            .map(|(id, _, team, key, _)| {
                BTreeMap::from([
                    ("request_id".into(), json!(format!("request-{id}"))),
                    ("response_id".into(), json!(id)),
                    ("litellm_call_id".into(), json!(id)),
                    ("team_id".into(), json!(team)),
                    ("api_key".into(), json!(key)),
                    ("start_time".into(), json!(start_ms)),
                    ("end_time".into(), json!(start_ms + 1)),
                    ("spend".into(), json!(0.25)),
                ])
            })
            .collect(),
    )
    .await?;
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into(), "team-b".into()],
    };
    let page = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    assert_eq!(page.data.len(), cases.len());
    let summary = page
        .data
        .iter()
        .find(|summary| summary.trace_id == id)
        .ok_or("missing run")?;
    let detail = reader
        .get_trace(&store, &access, id, &summary.trace_ref)
        .await?
        .ok_or("missing trace")?;
    assert_eq!(detail.summary.spend, expected, "{id}");
    assert_eq!(summary.spend, expected, "{id}");
    assert_eq!(summary.priced_calls, u64::from(expected.is_some()), "{id}");
    Ok(())
}

#[rstest]
#[case::provider_request(true)]
#[case::transport(false)]
#[tokio::test]
async fn native_cost_correlation_survives_session_grouping_and_excludes_other_owners(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] response_header: bool,
    #[values(false, true)] grouped: bool,
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let trace_id = if grouped {
        "grouped-trace"
    } else {
        "original-trace"
    };
    let start_ms = 1_790_000_000_000_i64;
    let keys = if response_header {
        vec!["provider_response:req_native"]
    } else {
        Vec::new()
    };
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!(start_ms * 1_000_000)),
            ("TraceId".into(), json!(trace_id)),
            ("SpanId".into(), json!("native-call")),
            ("SpanName".into(), json!("claude_code.llm_request")),
            ("ObservationType".into(), json!("llm")),
            ("Framework".into(), json!("claude-code")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("CallKeys".into(), json!(keys)),
            (
                "CallEvidence".into(),
                json!(if response_header {
                    "complete"
                } else {
                    "unknown"
                }),
            ),
            (
                "SpanAttributes".into(),
                json!({"lens.original_trace_id": if grouped { "original-trace" } else { "" }}),
            ),
        ])],
    )
    .await?;
    insert_rows(
        client,
        &writer,
        DATABASE,
        InsertTable::SpendLogs,
        ["key-a", "key-b"]
            .into_iter()
            .map(|key| {
                BTreeMap::from([
                    ("request_id".into(), json!(format!("log-{key}"))),
                    ("response_id".into(), json!("msg_native")),
                    ("provider_request_id".into(), json!("req_native")),
                    (
                        "trace_id".into(),
                        json!(if response_header {
                            ""
                        } else {
                            "original-trace"
                        }),
                    ),
                    (
                        "span_id".into(),
                        json!(if response_header { "" } else { "native-call" }),
                    ),
                    ("team_id".into(), json!("team-a")),
                    ("api_key".into(), json!(key)),
                    ("start_time".into(), json!(start_ms)),
                    ("end_time".into(), json!(start_ms + 1)),
                    ("spend".into(), json!(0.25)),
                ])
            })
            .collect(),
    )
    .await?;
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let page = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms: 0,
                end_ms: 2_000_000_000_000,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    assert_eq!(page.data.len(), 1);
    let summary = &page.data[0];
    let detail = reader
        .get_trace(&store, &access, trace_id, &summary.trace_ref)
        .await?
        .ok_or("missing trace")?;
    assert_eq!((summary.spend, summary.priced_calls), (Some(0.25), 1));
    assert_eq!(detail.summary.spend, summary.spend);
    assert_eq!(
        detail.spans[0].spend_log_request_id.as_deref(),
        Some("log-key-a")
    );
    Ok(())
}

const LABEL_START_MS: i64 = 1_790_000_000_000;

fn label_row(
    trace: &str,
    offset_ms: i64,
    (span, parent, name): (&str, &str, &str),
    kind: &str,
    (agent, service): (&str, &str),
    overrides: &[(&str, serde_json::Value)],
) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        (
            "Timestamp".into(),
            json!((LABEL_START_MS + offset_ms) * 1_000_000),
        ),
        ("TraceId".into(), json!(trace)),
        ("SpanId".into(), json!(span)),
        ("ParentSpanId".into(), json!(parent)),
        ("SpanName".into(), json!(name)),
        ("ObservationType".into(), json!(kind)),
        ("AgentName".into(), json!(agent)),
        ("WrapperCandidate".into(), json!(false)),
        ("ServiceName".into(), json!(service)),
        ("TeamId".into(), json!("team-a")),
        ("ApiKeyHash".into(), json!("key")),
        ("UserId".into(), json!("user-a")),
        ("Duration".into(), json!(1_000_000)),
    ])
    .into_iter()
    .chain(
        overrides
            .iter()
            .map(|(column, value)| ((*column).to_owned(), value.clone())),
    )
    .collect()
}

async fn seed_label_runs(fixture: &SeededDatabase) -> TestResult {
    let wrapper = [("WrapperCandidate", json!(true))];
    let other_team = [("TeamId", json!("team-b")), ("UserId", json!("user-b"))];
    let claude = ("claude-code", "claude-code");
    let rows = vec![
        label_row(
            "claude-run",
            1_000,
            ("c1", "", "claude_code.interaction"),
            "agent",
            claude,
            &[],
        ),
        label_row(
            "claude-run",
            1_001,
            ("c2", "c1", "claude_code.llm_request"),
            "llm",
            claude,
            &[],
        ),
        label_row(
            "named-run",
            2_000,
            ("n1", "", "invoke_agent researcher"),
            "agent",
            ("researcher", "research-app"),
            &[],
        ),
        label_row(
            "named-run",
            2_001,
            ("n2", "n1", "helper"),
            "agent",
            ("", "research-app"),
            &[],
        ),
        label_row(
            "unnamed-run",
            3_000,
            ("u1", "", "planner"),
            "agent",
            ("", "plan-app"),
            &[],
        ),
        label_row(
            "plain-run",
            4_000,
            ("p1", "", "POST /chat"),
            "llm",
            ("", "chat-service"),
            &[],
        ),
        label_row(
            "wrapped-run",
            5_000,
            ("w1", "", "Agent workflow"),
            "agent",
            ("", "wrap-app"),
            &wrapper,
        ),
        label_row(
            "wrapped-run",
            5_001,
            ("w2", "w1", "invoke_agent writer"),
            "agent",
            ("writer", "wrap-app"),
            &[],
        ),
        label_row(
            "lone-wrapper-run",
            6_000,
            ("l1", "", "lone wrapper"),
            "agent",
            ("", "lone-app"),
            &wrapper,
        ),
        label_row(
            "other-team-run",
            7_000,
            ("o1", "", "invoke_agent intruder"),
            "agent",
            ("intruder", "other"),
            &other_team,
        ),
        label_row(
            "old-run",
            -86_400_000,
            ("x1", "", "invoke_agent ancient"),
            "agent",
            ("ancient", "old-app"),
            &[],
        ),
        label_row(
            "newest-run",
            8_000,
            ("z1", "", "invoke_agent researcher"),
            "agent",
            ("researcher", "research-app"),
            &[],
        ),
    ];
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        rows,
    )
    .await?;
    Ok(())
}

async fn trace_agents(
    fixture: &SeededDatabase,
    connection: &Connection,
    access: &ReadAccessParams,
    end_ms: i64,
) -> TestResult<Vec<String>> {
    let parameters = BTreeMap::from([
        (
            "all_teams".into(),
            Parameter::Integer(i64::from(access.all_teams)),
        ),
        ("user_id".into(), Parameter::Text(access.user_id.clone())),
        (
            "team_ids".into(),
            Parameter::Strings(access.team_ids.clone()),
        ),
        ("start_ms".into(), Parameter::Integer(LABEL_START_MS)),
        ("end_ms".into(), Parameter::Integer(end_ms)),
        ("limit".into(), Parameter::Unsigned(1_000)),
    ]);
    let body = execute_named_read(
        &fixture.database.client,
        connection,
        ReadQuery::TraceAgents,
        &parameters,
    )
    .await?;
    let rows: serde_json::Value = serde_json::from_str(&body)?;
    Ok(rows["data"]
        .as_array()
        .ok_or("agent rows")?
        .iter()
        .map(|row| row["agent_name"].as_str().unwrap_or_default().to_owned())
        .collect())
}

fn label_access(all_teams: bool, user_id: &str, team_ids: &[&str]) -> ReadAccessParams {
    ReadAccessParams {
        all_teams,
        user_id: user_id.into(),
        team_ids: team_ids.iter().map(|team| (*team).to_owned()).collect(),
    }
}

fn trace_ids(page: &litellm_traces::TracePage) -> Vec<&str> {
    page.data.iter().map(|run| run.trace_id.as_str()).collect()
}

const TEAM_A_AGENTS: [&str; 7] = [
    "chat-service",
    "claude-code",
    "helper",
    "lone wrapper",
    "planner",
    "researcher",
    "writer",
];

#[rstest]
#[case::team(label_access(false, "", &["team-a"]), TEAM_A_AGENTS.to_vec())]
#[case::user(label_access(false, "user-a", &[]), TEAM_A_AGENTS.to_vec())]
#[case::all_teams(
    label_access(true, "", &[]),
    vec!["chat-service", "claude-code", "helper", "intruder", "lone wrapper", "planner", "researcher", "writer"]
)]
#[case::other_team(label_access(false, "", &["team-b"]), vec!["intruder"])]
#[tokio::test]
async fn trace_agents_lists_scoped_window_labels(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] access: ReadAccessParams,
    #[case] expected: Vec<&str>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_label_runs(&fixture).await?;
    let connection = fixture
        .readers
        .connection(&fixture.database.client, &QueryScope::All, "fixture-secret")
        .await?;
    let agents = trace_agents(&fixture, &connection, &access, LABEL_START_MS + 60_000).await?;
    assert_eq!(agents, expected);
    let before_runs = trace_agents(&fixture, &connection, &access, LABEL_START_MS + 500).await?;
    assert!(before_runs.is_empty(), "{before_runs:?}");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn list_traces_filters_by_agent_before_limit(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_label_runs(&fixture).await?;
    let connection = fixture
        .readers
        .connection(&fixture.database.client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(&fixture.database.client, connection);
    let access = label_access(false, "", &["team-a"]);
    let (start_ms, end_ms) = (LABEL_START_MS, LABEL_START_MS + 60_000);
    let list = |cursor: Option<String>, limit: u32, agent: &'static str| {
        let (reader, store, access) = (&reader, &store, &access);
        async move {
            reader
                .list_traces(
                    store,
                    access,
                    TraceListQuery {
                        start_ms,
                        end_ms,
                        cursor: cursor.as_deref(),
                        limit,
                        agent,
                    },
                )
                .await
        }
    };
    let first = list(None, 1, "researcher").await?;
    assert_eq!(trace_ids(&first), ["newest-run"]);
    let next = list(first.next_cursor.clone(), 1, "researcher").await?;
    assert_eq!(trace_ids(&next), ["named-run"]);
    assert_eq!(
        trace_ids(&list(None, 1, "claude-code").await?),
        ["claude-run"]
    );
    assert_eq!(
        trace_ids(&list(None, 10, "chat-service").await?),
        ["plain-run"]
    );
    assert_eq!(trace_ids(&list(None, 10, "helper").await?), ["named-run"]);
    assert!(list(None, 10, "intruder").await?.data.is_empty());
    assert!(list(None, 10, "Agent workflow").await?.data.is_empty());
    let unfiltered = list(None, 1, "").await?;
    assert_eq!(trace_ids(&unfiltered), ["newest-run"]);
    assert!(unfiltered.next_cursor.is_some());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn sql_agent_labels_match_resolved_run_summaries(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    seed_label_runs(&fixture).await?;
    let connection = fixture
        .readers
        .connection(&fixture.database.client, &QueryScope::All, "fixture-secret")
        .await?;
    let (reader, store) = make_reader(&fixture.database.client, connection.clone());
    let access = label_access(true, "", &[]);
    let (start_ms, end_ms) = (LABEL_START_MS, LABEL_START_MS + 60_000);
    let page = reader
        .list_traces(
            &store,
            &access,
            TraceListQuery {
                start_ms,
                end_ms,
                cursor: None,
                limit: 50,
                agent: "",
            },
        )
        .await?;
    let agents = trace_agents(&fixture, &connection, &access, end_ms).await?;
    let listed = fetch::<ListTraces>(
        &fixture.database.client,
        &connection,
        &ListTracesParams::from(contracts::ListTracesParams {
            access: access.clone(),
            start_ms,
            end_ms,
            cursor_ms: 0,
            cursor_trace_id: String::new(),
            limit: 50,
            agent: String::new(),
        }),
    )
    .await?;
    assert_eq!(page.data.len(), 8);
    let mut filtered = BTreeMap::new();
    for agent in &agents {
        let runs = reader
            .list_traces(
                &store,
                &access,
                TraceListQuery {
                    start_ms,
                    end_ms,
                    cursor: None,
                    limit: 50,
                    agent,
                },
            )
            .await?;
        filtered.insert(agent.as_str(), runs);
    }
    for summary in &page.data {
        assert!(!summary.resolution_limited, "{}", summary.trace_id);
        let labels = if summary.agent_names.is_empty() {
            vec![summary.service.clone()]
        } else {
            summary.agent_names.clone()
        };
        let row = listed
            .iter()
            .find(|row| row.0.trace_id == summary.trace_id)
            .ok_or("missing listed row")?;
        assert_eq!(
            row.0.agent_names, summary.agent_names,
            "{}",
            summary.trace_id
        );
        assert!(
            labels.iter().all(|label| agents.contains(label)),
            "{labels:?}"
        );
        for (agent, runs) in &filtered {
            assert_eq!(
                trace_ids(runs).contains(&summary.trace_id.as_str()),
                labels.iter().any(|label| label == agent),
                "{} under {agent}",
                summary.trace_id
            );
        }
    }
    Ok(())
}
