use std::collections::BTreeMap;

use litellm_storage_clickhouse::fetch;
use litellm_traces::query::named as contracts;
use litellm_traces_clickhouse::{
    QueryScope,
    query::named::{ListTraces, ListTracesParams, TraceSpans, TraceSpansParams},
    query_sql,
};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::Value;

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{SeededDatabase, insert_export, migrated_database, seeded_database};
use support::TestResult;

#[derive(Clone, Copy, strum::AsRefStr)]
#[strum(serialize_all = "snake_case")]
enum ScopeCase {
    Admin,
    Team,
    Key,
    OtherTeam,
}

impl ScopeCase {
    fn scope(self) -> QueryScope {
        match self {
            Self::Admin => QueryScope::Admin,
            Self::Team => QueryScope::Team {
                team_id: "team-a".into(),
            },
            Self::Key => QueryScope::Key {
                team_id: "team-a".into(),
                api_key_hash: "key-a".into(),
            },
            Self::OtherTeam => QueryScope::Team {
                team_id: "team-b".into(),
            },
        }
    }
}

#[derive(Deserialize)]
struct QueryResult {
    data: Vec<Value>,
}

#[rstest]
#[case::rollups(include_str!("queries/rollups.sql"), include_str!("queries/rollups.expected.json"))]
#[case::costs(include_str!("queries/trace_costs.sql"), include_str!("queries/trace_costs.expected.json"))]
#[case::errors(include_str!("queries/failed_spans.sql"), include_str!("queries/failed_spans.expected.json"))]
#[case::metadata(include_str!("queries/metadata_filters.sql"), include_str!("queries/metadata_filters.expected.json"))]
#[tokio::test]
async fn curated_queries_return_expected_rows(
    #[future(awt)] seeded_database: TestResult<SeededDatabase>,
    #[case] sql: &str,
    #[case] expected_json: &str,
    #[values(
        ScopeCase::Admin,
        ScopeCase::Team,
        ScopeCase::Key,
        ScopeCase::OtherTeam
    )]
    scope: ScopeCase,
) -> TestResult {
    let fixture = seeded_database?;
    let reader = fixture
        .readers
        .connection(&fixture.database.client, &scope.scope(), "fixture-secret")
        .await?;
    let result: QueryResult =
        serde_json::from_str(&query_sql(&fixture.database.client, &reader, sql).await?)?;
    let expected: BTreeMap<String, Vec<Value>> = serde_json::from_str(expected_json)?;
    assert_eq!(
        &result.data,
        expected
            .get(scope.as_ref())
            .ok_or("missing expected scope")?,
        "{}: {sql}",
        scope.as_ref()
    );
    Ok(())
}

#[fixture]
fn admin_access() -> TestResult<contracts::ReadAccessParams> {
    Ok(serde_json::from_str(include_str!(
        "queries/read_access.json"
    ))?)
}

#[rstest]
#[tokio::test]
async fn typed_queries_read_normalized_spans_and_keep_trace_identities_separate(
    #[future(awt)] seeded_database: TestResult<SeededDatabase>,
    admin_access: TestResult<contracts::ReadAccessParams>,
) -> TestResult {
    let fixture = seeded_database?;
    let reader = fixture
        .readers
        .connection(
            &fixture.database.client,
            &QueryScope::Admin,
            "fixture-secret",
        )
        .await?;
    let params = ListTracesParams::from(contracts::ListTracesParams {
        access: admin_access?,
        start_ms: 0,
        end_ms: i64::MAX / 1_000_000,
        cursor_ms: 0,
        cursor_trace_id: String::new(),
        limit: 10,
    });
    let traces = fetch::<ListTraces>(&fixture.database.client, &reader, &params).await?;
    assert_eq!(
        traces
            .iter()
            .map(|row| row.0.api_key_hash.as_str())
            .collect::<Vec<_>>(),
        ["key-b", "key-alt", "key-a"]
    );
    let trace = &traces[2].0;
    assert_eq!(
        (
            trace.span_count,
            trace.llm_calls,
            trace.tool_calls,
            trace.error_count
        ),
        (3, 1, 1, 1)
    );
    assert_eq!((trace.input_tokens, trace.output_tokens), (12, 6));
    assert_eq!(trace.input_preview, "Review the change");
    let span_params = TraceSpansParams {
        access: params.0.access,
        trace_id: trace.trace_id.clone(),
        trace_ref: trace.trace_ref.clone(),
    };
    let spans = fetch::<TraceSpans>(&fixture.database.client, &reader, &span_params).await?;
    assert_eq!(
        spans
            .iter()
            .map(|row| row.0.name.as_str())
            .collect::<Vec<_>>(),
        ["review", "completion", "lookup"]
    );
    assert!(
        spans
            .iter()
            .all(|row| row.0.api_key_hash == trace.api_key_hash)
    );
    assert_eq!(
        (
            spans[1].0.kind.as_str(),
            spans[1].0.input_tokens,
            spans[1].0.output_tokens
        ),
        ("llm", 12, 6)
    );
    assert_eq!(spans[2].0.status_message, "lookup timed out");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn typed_trace_cursor_returns_the_next_fixture_trace(
    #[future(awt)] seeded_database: TestResult<SeededDatabase>,
    admin_access: TestResult<contracts::ReadAccessParams>,
) -> TestResult {
    let fixture = seeded_database?;
    let reader = fixture
        .readers
        .connection(
            &fixture.database.client,
            &QueryScope::Admin,
            "fixture-secret",
        )
        .await?;
    let params = ListTracesParams::from(contracts::ListTracesParams {
        access: admin_access?,
        start_ms: 0,
        end_ms: i64::MAX / 1_000_000,
        cursor_ms: 0,
        cursor_trace_id: String::new(),
        limit: 1,
    });
    let first = fetch::<ListTraces>(&fixture.database.client, &reader, &params).await?;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0.api_key_hash, "key-b");
    let next_params = ListTracesParams::from(contracts::ListTracesParams {
        cursor_ms: first[0].0.start_ms,
        cursor_trace_id: first[0].0.trace_ref.clone(),
        ..params.0
    });
    let next = fetch::<ListTraces>(&fixture.database.client, &reader, &next_params).await?;
    assert_eq!(next.len(), 1);
    assert_eq!(next[0].0.api_key_hash, "key-alt");
    assert_ne!(first[0].0.trace_ref, next[0].0.trace_ref);
    Ok(())
}

#[rstest]
#[case::authentication_error(include_bytes!("../../traces/tests/fixtures/deeplite_auth_error.json"))]
#[case::swarm(include_bytes!("../../traces/tests/fixtures/deeplite_swarm.json"))]
#[tokio::test]
async fn captured_deeplite_exports_round_trip_through_clickhouse(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    admin_access: TestResult<contracts::ReadAccessParams>,
    #[case] export: &[u8],
) -> TestResult {
    let fixture = migrated_database?;
    let decoded = insert_export(&fixture, export, "team-a", "key-a").await?;
    let reader = fixture
        .readers
        .connection(
            &fixture.database.client,
            &QueryScope::Admin,
            "fixture-secret",
        )
        .await?;
    let params = TraceSpansParams {
        access: admin_access?,
        trace_id: decoded[0].trace_id.clone(),
        trace_ref: String::new(),
    };
    let stored = fetch::<TraceSpans>(&fixture.database.client, &reader, &params).await?;
    assert_eq!(stored.len(), decoded.len());
    let list_params = ListTracesParams::from(contracts::ListTracesParams {
        access: params.access,
        start_ms: 0,
        end_ms: i64::MAX / 1_000_000,
        cursor_ms: 0,
        cursor_trace_id: String::new(),
        limit: 10,
    });
    let traces = fetch::<ListTraces>(&fixture.database.client, &reader, &list_params).await?;
    assert_eq!(traces.len(), 1);
    let roots = decoded
        .iter()
        .filter(|span| span.parent_span_id.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), 1);
    assert_eq!(traces[0].0.status, roots[0].status_code);
    assert_eq!(
        traces[0].0.error_count,
        decoded
            .iter()
            .filter(|span| span.status_code == "STATUS_CODE_ERROR")
            .count() as u64
    );
    let by_id: BTreeMap<_, _> = stored
        .iter()
        .map(|row| (row.0.span_id.as_str(), &row.0))
        .collect();
    for span in &decoded {
        let row = by_id
            .get(span.span_id.as_str())
            .ok_or("missing captured span")?;
        assert_eq!(row.parent_span_id, span.parent_span_id);
        assert_eq!(row.start_ns as u64, span.start_ns);
        assert_eq!(row.duration_ns, span.end_ns - span.start_ns);
        assert_eq!(row.input_tokens, span.normalized.input_tokens);
        assert_eq!(row.output_tokens, span.normalized.output_tokens);
        assert_eq!(row.status, span.status_code);
    }
    Ok(())
}
