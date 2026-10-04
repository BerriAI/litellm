use std::collections::{BTreeMap, BTreeSet};

use litellm_traces::search::{AgentRuns, HistogramBucket, RunField, RunFilter, RunSearch};
use litellm_traces_cache::{PageRequest, TraceReader};
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, InsertTable, QueryScope, insert_rows,
};
use rstest::rstest;
use serde_json::{Value, json};
use strum::IntoEnumIterator;

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database};
use support::TestResult;

const HOUR_MS: i64 = 3_600_000;
const T0_MS: i64 = 1_790_000_000_000;
const WINDOW_END_MS: i64 = T0_MS + 3 * HOUR_MS;

type ExpectedBucket = (u64, u64, Vec<(&'static str, u64)>);

fn filter(start_ms: i64, q: &str) -> RunFilter {
    RunFilter {
        start_ms,
        end_ms: WINDOW_END_MS,
        search: RunSearch::parse(q),
    }
}

fn page(cursor: Option<String>, limit: u32) -> PageRequest {
    PageRequest { cursor, limit }
}

fn reader() -> TraceReader {
    TraceReader::new(litellm_storage_clickhouse::READ_LIMITS.response_bytes)
}

struct Step {
    span_id: &'static str,
    name: &'static str,
    kind: &'static str,
    agent: &'static str,
    model: &'static str,
    error: bool,
}

const fn step(span_id: &'static str, name: &'static str, kind: &'static str) -> Step {
    Step {
        span_id,
        name,
        kind,
        agent: "",
        model: "",
        error: false,
    }
}

struct Run {
    trace_id: &'static str,
    team: &'static str,
    start_ms: i64,
    root: &'static str,
    input: &'static str,
    steps: Vec<Step>,
}

fn runs() -> Vec<Run> {
    vec![
        Run {
            trace_id: "alpha",
            team: "team-a",
            start_ms: T0_MS,
            root: "plan trip",
            input: "book a flight to Paris",
            steps: vec![
                Step {
                    agent: "researcher",
                    ..step("alpha-agent", "invoke_agent", "agent")
                },
                Step {
                    model: "gpt-x",
                    ..step("alpha-llm", "chat", "llm")
                },
            ],
        },
        Run {
            trace_id: "beta",
            team: "team-a",
            start_ms: T0_MS + HOUR_MS,
            root: "write report",
            input: "summarize 100% of Q3_sales",
            steps: vec![
                step("beta-agent", "writer", "agent"),
                Step {
                    model: "claude-y",
                    error: true,
                    ..step("beta-llm", "chat", "llm")
                },
            ],
        },
        Run {
            trace_id: "gamma",
            team: "team-a",
            start_ms: T0_MS + 2 * HOUR_MS,
            root: "plan trip",
            input: "hello",
            steps: vec![Step {
                model: "gpt-x",
                ..step("gamma-llm", "chat", "llm")
            }],
        },
        Run {
            trace_id: "foreign",
            team: "team-b",
            start_ms: T0_MS + 30 * 60_000,
            root: "plan trip",
            input: "book a flight to Paris",
            steps: vec![Step {
                agent: "spy",
                model: "gpt-x",
                ..step("foreign-agent", "invoke_agent", "agent")
            }],
        },
    ]
}

fn span_row(run: &Run, offset_ms: i64, span: &Step, parent: &str) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "Timestamp".into(),
            json!((run.start_ms + offset_ms) * 1_000_000),
        ),
        ("TraceId".into(), json!(run.trace_id)),
        ("SpanId".into(), json!(span.span_id)),
        ("ParentSpanId".into(), json!(parent)),
        ("SpanName".into(), json!(span.name)),
        ("ServiceName".into(), json!("svc")),
        ("ObservationType".into(), json!(span.kind)),
        ("AgentName".into(), json!(span.agent)),
        ("Model".into(), json!(span.model)),
        (
            "StatusCode".into(),
            json!(if span.error {
                "STATUS_CODE_ERROR"
            } else {
                "STATUS_CODE_OK"
            }),
        ),
        ("TeamId".into(), json!(run.team)),
        ("ApiKeyHash".into(), json!("key-a")),
        ("Duration".into(), json!(1_000_000)),
    ])
}

fn rows(run: &Run) -> Vec<BTreeMap<String, Value>> {
    let root = step("root", run.root, "chain");
    let root_id = format!("{}-root", run.trace_id);
    let root_row = BTreeMap::from_iter(span_row(run, 0, &root, "").into_iter().chain([
        ("SpanId".into(), json!(root_id)),
        ("Input".into(), json!(run.input)),
    ]));
    std::iter::once(root_row)
        .chain(
            run.steps
                .iter()
                .zip(1..)
                .map(|(span, offset)| span_row(run, offset, span, &root_id)),
        )
        .collect()
}

async fn seed(fixture: &SeededDatabase) -> TestResult<ClickHouseTraces> {
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        runs().iter().flat_map(rows).collect(),
    )
    .await?;
    let connection = fixture
        .readers
        .connection(&fixture.database.client, &QueryScope::All, "fixture-secret")
        .await?;
    Ok(ClickHouseTraces::new(
        fixture.database.client.clone(),
        connection,
    ))
}

fn team_a() -> QueryScope {
    QueryScope::Owned {
        user_id: String::new(),
        team_ids: vec!["team-a".into()],
    }
}

#[rstest]
#[tokio::test]
async fn list_q_selects_matching_runs_before_paging(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let reader = reader();
    let cases: &[(&str, &[&str])] = &[
        ("", &["gamma", "beta", "alpha"]),
        ("status:", &["gamma", "beta", "alpha"]),
        (r#"name:"plan trip""#, &["gamma", "alpha"]),
        (r#"-name:"plan trip""#, &["beta"]),
        ("NAME:PLAN*", &["gamma", "alpha"]),
        ("agent:researcher", &["alpha"]),
        ("agent:writer", &["beta"]),
        ("agent:RESEARCH*", &["alpha"]),
        ("agent:research", &[]),
        ("-agent:*", &["gamma"]),
        ("status:error", &["beta"]),
        ("status:ok", &["gamma", "alpha"]),
        ("status:*r*", &["beta"]),
        ("model:gpt-x", &["gamma", "alpha"]),
        ("model:gpt", &[]),
        ("-model:gpt-x", &["beta"]),
        ("input:*paris", &["alpha"]),
        ("trace_id:gam*", &["gamma"]),
        ("paris", &["alpha"]),
        ("PLAN", &["gamma", "alpha"]),
        ("100%", &["beta"]),
        ("summarize%sales", &[]),
        (r#""100% of""#, &["beta"]),
        (r#""100%  of""#, &[]),
        ("q3_", &["beta"]),
        ("flight_to", &[]),
        ("plan hello", &["gamma"]),
        ("plan -trace_id:gamma", &["alpha"]),
        ("unknown:x", &[]),
    ];
    for (q, expected) in cases {
        let page = reader
            .list_traces(&store, &team_a(), &filter(0, q), &page(None, 50))
            .await?;
        let listed: Vec<&str> = page.data.iter().map(|run| run.trace_id.as_str()).collect();
        assert_eq!(&listed, expected, "q = {q:?}");
    }
    Ok(())
}

#[rstest]
#[tokio::test]
async fn list_q_pages_through_matches_only(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let reader = reader();
    let filter = filter(0, "model:gpt-x");
    let first = reader
        .list_traces(&store, &team_a(), &filter, &page(None, 1))
        .await?;
    let second = reader
        .list_traces(
            &store,
            &team_a(),
            &filter,
            &page(first.next_cursor.clone(), 1),
        )
        .await?;
    let ids = |page: &litellm_traces::TracePage| -> Vec<String> {
        page.data.iter().map(|run| run.trace_id.clone()).collect()
    };
    assert_eq!(ids(&first), ["gamma"]);
    assert_eq!(ids(&second), ["alpha"]);
    Ok(())
}

#[rstest]
#[case::all("", [
    (1, 0, vec![("researcher", 1)]),
    (1, 1, vec![]),
    (1, 0, vec![("svc", 1)]),
])]
#[case::filtered("model:gpt-x", [
    (1, 0, vec![("researcher", 1)]),
    (0, 0, vec![]),
    (1, 0, vec![("svc", 1)]),
])]
#[tokio::test]
async fn histogram_counts_matching_runs_per_bucket(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] q: &str,
    #[case] expected: [ExpectedBucket; 3],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let histogram = reader()
        .histogram(&store, &team_a(), &filter(T0_MS, q), 3)
        .await?;
    let expected: Vec<HistogramBucket> = expected
        .into_iter()
        .zip(0..)
        .map(|((total, failed, agents), index)| HistogramBucket {
            start_ms: T0_MS + index * HOUR_MS,
            end_ms: T0_MS + (index + 1) * HOUR_MS,
            total,
            failed,
            agents: agents
                .into_iter()
                .map(|(agent, runs)| AgentRuns {
                    agent: agent.into(),
                    runs,
                })
                .collect(),
        })
        .collect();
    assert_eq!(histogram.buckets, expected);
    Ok(())
}

#[rstest]
#[case::agents_in_scope(RunField::Agent, "", &["researcher", "writer"])]
#[case::names_by_frequency(RunField::Name, "", &["plan trip", "write report"])]
#[case::statuses_by_frequency(RunField::Status, "", &["ok", "error"])]
#[case::models_by_frequency(RunField::Model, "", &["gpt-x", "claude-y"])]
#[case::needle_ignores_case(RunField::Name, "WR", &["write report"])]
#[case::needle_matches_inside(RunField::Name, "trip", &["plan trip"])]
#[case::needle_underscore_is_literal(RunField::Name, "n_t", &[])]
#[case::needle_percent_is_literal(RunField::Name, "n%t", &[])]
#[tokio::test]
async fn values_list_distinct_field_values_in_scope(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] field: RunField,
    #[case] needle: &str,
    #[case] expected: &[&str],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let values = reader()
        .values(&store, &team_a(), &filter(0, ""), field, needle, 10)
        .await?;
    assert_eq!(values.values, expected);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn admin_scope_sees_every_team(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let admin = QueryScope::All;
    let values = reader()
        .values(&store, &admin, &filter(0, ""), RunField::Agent, "", 10)
        .await?;
    assert_eq!(
        values.values.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["researcher".into(), "spy".into(), "writer".into()])
    );
    Ok(())
}

#[rstest]
#[case::by_model(RunField::Agent, "model:claude-y", &["writer"])]
#[case::by_status(RunField::Name, "status:error", &["write report"])]
#[case::excluding(RunField::Model, "-trace_id:alpha", &["claude-y", "gpt-x"])]
#[tokio::test]
async fn values_narrow_to_runs_matching_the_search(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] field: RunField,
    #[case] q: &str,
    #[case] expected: &[&str],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let values = reader()
        .values(&store, &team_a(), &filter(0, q), field, "", 10)
        .await?;
    assert_eq!(values.values, expected);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn every_field_suggests_values_that_filter_back_to_their_runs(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let reader = reader();
    for field in RunField::iter() {
        let key: &str = field.into();
        let values = reader
            .values(&store, &team_a(), &filter(0, ""), field, "", 10)
            .await?;
        assert!(!values.values.is_empty(), "{key} suggests nothing");
        for value in &values.values {
            let q = format!(r#"{key}:"{value}""#);
            let included = reader
                .list_traces(&store, &team_a(), &filter(0, &q), &page(None, 50))
                .await?;
            let excluded = reader
                .list_traces(
                    &store,
                    &team_a(),
                    &filter(0, &format!("-{q}")),
                    &page(None, 50),
                )
                .await?;
            assert!(!included.data.is_empty(), "{q} lists nothing");
            assert_eq!(
                included.data.len() + excluded.data.len(),
                3,
                "{q} does not partition the runs"
            );
        }
    }
    Ok(())
}
