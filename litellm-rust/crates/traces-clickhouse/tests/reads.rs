use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch};
use litellm_traces::query::named::{self as contracts, ReadAccessParams};
use litellm_traces_cache::{ReadError, TraceReader, TraceStore};
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, InsertTable, insert_rows,
    query::named::{SpendByResponseIds, SpendByResponseIdsParams, SpendByResponseIdsRow},
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: user_id.into(),
        team_ids: vec!["team-a".into()],
    };
    let page = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into()],
    };
    let page = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 500)
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
        for table in ["otel_traces AS o", "spend_logs"]
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection.clone());
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let listed = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 10)
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection.clone());
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let before = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
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
            ("SpanName".into(), json!("x".repeat(16 * 1024 * 1024 + 1))),
            ("ObservationType".into(), json!("tool")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
        ])],
    )
    .await?;
    let cached = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
        .await?;
    assert_eq!(cached.data, before.data);
    let (reader, store) = make_reader(client, connection);
    let after = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into(), "team-b".into()],
    };
    let page = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
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
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let (reader, store) = make_reader(client, connection);
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let page = reader
        .list_traces(&store, &access, 0, 2_000_000_000_000, None, 50)
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

struct FinalSpend;

impl Query for FinalSpend {
    type Params = SpendByResponseIdsParams;
    type Row = SpendByResponseIdsRow;
    const SQL: &'static str = include_str!("queries/spend_final.sql");
}

fn spend_version(
    request_id: &str,
    fields: &[(&str, &str)],
    end_offset_ms: i64,
    cost: f64,
) -> BTreeMap<String, serde_json::Value> {
    let start_ms = 1_790_000_000_000_i64;
    BTreeMap::from([
        ("team_id".to_owned(), json!("team-a")),
        ("user".to_owned(), json!("user-a")),
        ("api_key".to_owned(), json!("key-a")),
    ])
    .into_iter()
    .chain(
        fields
            .iter()
            .map(|(name, value)| ((*name).to_owned(), json!(value))),
    )
    .chain([
        ("request_id".to_owned(), json!(request_id)),
        ("start_time".to_owned(), json!(start_ms)),
        ("end_time".to_owned(), json!(start_ms + end_offset_ms)),
        ("spend".to_owned(), json!(cost)),
    ])
    .collect()
}

fn sorted_spend(rows: Vec<contracts::SpendByResponseIdsRow>) -> TestResult<Vec<serde_json::Value>> {
    let mut values = rows
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by_key(|row| row["request_id"].to_string());
    Ok(values)
}

#[rstest]
#[case::all_teams(true, "", &[], &[("foreign", 4.0), ("reassigned", 2.0)])]
#[case::owned_by_user(false, "user-a", &[], &[])]
#[case::team_member(false, "", &["team-a"], &[("reassigned", 2.0)])]
#[tokio::test]
async fn spend_lookup_matches_the_final_read_including_replaced_versions(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] all_teams: bool,
    #[case] user_id: &str,
    #[case] team_ids: &[&str],
    #[case] scoped: &[(&str, f64)],
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let managed = format!(
        "resp_{}",
        STANDARD
            .encode("litellm:custom_llm_provider:openai;model_id:m;response_id:chatcmpl-upstream")
    );
    let inserts = [
        vec![
            spend_version("versioned", &[("response_id", "resp-versioned")], 1, 1.0),
            spend_version("tied", &[("trace_id", "trace-a")], 2, 1.0),
            spend_version("moved-away", &[("response_id", "resp-moved")], 1, 1.0),
            spend_version("moved-in", &[("response_id", "unrelated")], 1, 1.0),
            spend_version("managed", &[("response_id", managed.as_str())], 1, 0.5),
            spend_version(
                "provider",
                &[("provider_request_id", "req-provider")],
                1,
                0.75,
            ),
            spend_version("gateway", &[("litellm_call_id", "call-a")], 1, 0.125),
            spend_version("legacy-call", &[], 1, 0.0625),
            spend_version("claimed-call", &[("litellm_call_id", "other-call")], 1, 9.0),
            spend_version("empty-trace", &[("trace_id", "")], 1, 9.0),
            spend_version("unrelated", &[("response_id", "nothing")], 1, 9.0),
            spend_version("reassigned", &[("response_id", "resp-versioned")], 1, 1.0),
            spend_version(
                "foreign",
                &[
                    ("response_id", "resp-versioned"),
                    ("team_id", "team-b"),
                    ("user", "user-b"),
                ],
                1,
                4.0,
            ),
        ],
        vec![
            spend_version("versioned", &[("response_id", "resp-versioned")], 5, 2.0),
            spend_version("tied", &[("trace_id", "trace-a")], 2, 2.0),
            spend_version("moved-away", &[("response_id", "unrelated")], 5, 2.0),
            spend_version("moved-in", &[("response_id", "resp-moved-in")], 5, 2.0),
            spend_version(
                "reassigned",
                &[("response_id", "resp-versioned"), ("user", "user-b")],
                5,
                2.0,
            ),
        ],
        vec![
            spend_version("versioned", &[("response_id", "resp-versioned")], 3, 3.0),
            spend_version("tied", &[("trace_id", "trace-a")], 2, 3.0),
        ],
    ];
    for rows in inserts {
        insert_rows(client, &writer, DATABASE, InsertTable::SpendLogs, rows).await?;
    }
    let params = contracts::SpendByResponseIdsParams {
        access: ReadAccessParams {
            all_teams,
            user_id: user_id.into(),
            team_ids: team_ids.iter().map(|team| (*team).to_owned()).collect(),
        },
        response_ids: [
            "resp-versioned",
            "resp-moved",
            "resp-moved-in",
            "chatcmpl-upstream",
        ]
        .map(String::from)
        .to_vec(),
        provider_request_ids: vec!["req-provider".into()],
        request_ids: ["call-a", "legacy-call", "claimed-call"]
            .map(String::from)
            .to_vec(),
        trace_ids: vec!["trace-a".into(), String::new()],
        start_ms: 1_789_999_999_000,
        end_ms: 1_790_000_001_000,
    };
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let named_params = SpendByResponseIdsParams::from(params.clone());
    let expected = fetch::<FinalSpend>(client, &connection, &named_params).await?;
    let named = fetch::<SpendByResponseIds>(client, &connection, &named_params).await?;
    let batch = ClickHouseTraces::new(client.clone(), connection)
        .spend(&params)
        .await?;
    let costs: BTreeMap<_, _> = batch
        .iter()
        .map(|row| (row.request_id.clone(), row.spend))
        .collect();
    let visible = [
        ("gateway", 0.125),
        ("legacy-call", 0.0625),
        ("managed", 0.5),
        ("moved-in", 2.0),
        ("provider", 0.75),
        ("tied", 3.0),
        ("versioned", 2.0),
    ]
    .into_iter()
    .chain(scoped.iter().copied())
    .map(|(id, cost)| (id.to_owned(), Some(cost)))
    .collect::<BTreeMap<_, _>>();
    assert_eq!(costs, visible);
    let expected = sorted_spend(expected.into_iter().map(|row| row.0).collect())?;
    assert_eq!(sorted_spend(batch)?, expected);
    assert_eq!(
        sorted_spend(named.into_iter().map(|row| row.0).collect())?,
        expected
    );
    Ok(())
}
