use std::collections::BTreeMap;

use litellm_traces::query::named::ReadAccessParams;
use litellm_traces_clickhouse::{
    Connection, InsertTable, QueryScope, get_trace, get_trace_page, insert_rows, list_traces,
};
use rstest::rstest;
use serde_json::json;

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database, seeded_database};
use support::TestResult;

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
    let reader = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let access = ReadAccessParams {
        all_teams: false,
        user_id: user_id.into(),
        team_ids: vec!["team-a".into()],
    };
    let page = list_traces(client, &reader, &access, 0, 2_000_000_000_000, None, 50).await?;
    assert_eq!(page.data.len(), runs.len());
    for (trace_id, _, cost) in runs {
        let summary = page
            .data
            .iter()
            .find(|summary| summary.trace_id == trace_id)
            .ok_or("missing run")?;
        let detail = get_trace(client, &reader, &access, trace_id, &summary.trace_ref)
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
    let reader = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let access = ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into()],
    };
    let page = list_traces(client, &reader, &access, 0, 2_000_000_000_000, None, 500).await?;
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
        let read_queries = client.post(writer.url().clone()).body(format!(
            "SELECT count() FROM system.query_log WHERE type = 'QueryFinish' AND current_database = '{DATABASE}' AND query LIKE '%FROM otel_traces AS o%' AND query NOT LIKE '%system.query_log%'"
        )).send().await?.error_for_status()?.text().await?;
        let read_queries = read_queries.trim().parse::<usize>()?;
        assert!(
            read_queries > 0 && read_queries < runs,
            "{read_queries} span queries for {runs} runs"
        );
        if costed {
            let overlapping = client
                .post(writer.url().clone())
                .body(format!(
                    "WITH spend_reads AS (
                        SELECT query_start_time_microseconds AS started, event_time_microseconds AS finished
                        FROM system.query_log
                        WHERE type = 'QueryFinish' AND current_database = '{DATABASE}'
                          AND query LIKE '%FROM spend_logs FINAL%' AND query NOT LIKE '%system.query_log%'
                    ), events AS (
                        SELECT started AS at, 1 AS delta FROM spend_reads
                        UNION ALL SELECT finished AS at, -1 AS delta FROM spend_reads
                    )
                    SELECT max(active) FROM (
                        SELECT sum(delta) OVER (ORDER BY at, delta ROWS UNBOUNDED PRECEDING) AS active
                        FROM events
                    )"
                ))
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?
                .trim()
                .parse::<usize>()?;
            assert!(
                (2..=4).contains(&overlapping),
                "{overlapping} simultaneous spend reads for {runs} runs"
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
    let detail = get_trace(client, &reader, &access, "trace-0000", trace_ref)
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
        get_trace(client, &reader, &denied, "trace-0000", trace_ref)
            .await?
            .is_none()
    );
    let mut cursor = None;
    let mut ids = Vec::new();
    loop {
        let page = get_trace_page(
            client,
            &reader,
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
                get_trace_page(
                    client,
                    &reader,
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
    let reader = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let listed = list_traces(client, &reader, &access, 0, 2_000_000_000_000, None, 10).await?;
    let summary = listed
        .data
        .iter()
        .find(|summary| summary.span_count == 3)
        .ok_or("missing fixture")?;
    let first = get_trace_page(
        client,
        &reader,
        &access,
        &summary.trace_id,
        &summary.trace_ref,
        None,
        1,
    )
    .await?
    .ok_or("missing first page")?;
    let original_ids = get_trace(
        client,
        &reader,
        &access,
        &summary.trace_id,
        &summary.trace_ref,
    )
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
        get_trace_page(
            client,
            &reader,
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
        let next = get_trace_page(
            client,
            &reader,
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
    let refreshed = get_trace(
        client,
        &reader,
        &access,
        &summary.trace_id,
        &summary.trace_ref,
    )
    .await?
    .ok_or("missing refreshed trace")?;
    assert_eq!(refreshed.spans.len(), 4);
    assert!(matches!(
        get_trace_page(
            client,
            &reader,
            &access,
            &summary.trace_id,
            &summary.trace_ref,
            Some("invalid"),
            1
        )
        .await,
        Err(litellm_traces_clickhouse::Error::InvalidCursor("span"))
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
    let uncached_reader =
        Connection::reader(&format!("{}?max_threads=1", fixture.database.url), DATABASE)?;
    let changed = get_trace_page(
        client,
        &uncached_reader,
        &access,
        &summary.trace_id,
        &summary.trace_ref,
        first_cursor.as_deref(),
        1,
    )
    .await;
    assert!(
        matches!(changed, Err(litellm_traces_clickhouse::Error::TraceChanged)),
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
    let reader = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let access = ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    };
    let before = list_traces(client, &reader, &access, 0, 2_000_000_000_000, None, 50).await?;
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
    let after = list_traces(client, &reader, &access, 0, 2_000_000_000_000, None, 50).await?;
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
        get_trace_page(
            client,
            &reader,
            &access,
            &run.trace_id,
            &run.trace_ref,
            None,
            200
        )
        .await,
        Err(litellm_traces_clickhouse::Error::ReadTooLarge)
    ));
    Ok(())
}
