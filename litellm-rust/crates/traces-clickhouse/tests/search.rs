use std::collections::{BTreeMap, BTreeSet};

use litellm_traces::{
    search::{AgentRuns, HistogramBucket, RunField, RunFilter, RunSearch},
    store::{
        CountBy, CountValue, RunCountQuery, RunOrder, RunQuery, RunSelection, RunSortKey, SpanPart,
        SpanQuery, SpanSelection, TextRange,
    },
};
use litellm_traces_cache::{PageRequest, TraceReader, TraceStore};
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, InsertTable, QueryScope, encode_rows, insert_rows,
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
        search: RunSearch::parse(q).unwrap(),
        ..Default::default()
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
        (r#"name:"plan trip""#, &["gamma", "alpha"]),
        (r#"-name:"plan trip""#, &["beta"]),
        ("NAME:PLAN*", &["gamma", "alpha"]),
        ("agent:researcher", &["alpha"]),
        ("agent:writer", &["beta"]),
        ("agent:RESEARCH*", &["alpha"]),
        ("agent:research", &[]),
        ("-agent:*", &["gamma"]),
        ("has_error:true", &["beta"]),
        ("has_error:false", &["gamma", "alpha"]),
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
    ];
    for (q, expected) in cases {
        let page = reader
            .list_traces(
                &store,
                &team_a(),
                &filter(0, q),
                RunOrder::NEWEST,
                &page(None, 50),
            )
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
        .list_traces(&store, &team_a(), &filter, RunOrder::NEWEST, &page(None, 1))
        .await?;
    let second = reader
        .list_traces(
            &store,
            &team_a(),
            &filter,
            RunOrder::NEWEST,
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
#[tokio::test]
async fn named_and_unnamed_agents_in_one_run_remain_searchable(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![span_row(
            &runs()[0],
            5,
            &step("alpha-unnamed", "helper", "agent"),
            "alpha-root",
        )],
    )
    .await?;
    let reader = reader();
    let values = reader
        .values(
            &store,
            &team_a(),
            &filter(T0_MS, "trace_id:alpha"),
            RunField::Agent,
            "",
            10,
        )
        .await?;
    assert_eq!(values.values, ["helper", "researcher"]);
    for agent in values.values {
        let listed = reader
            .list_traces(
                &store,
                &team_a(),
                &filter(T0_MS, &format!("agent:{agent}")),
                RunOrder::NEWEST,
                &page(None, 10),
            )
            .await?;
        assert_eq!(listed.data.len(), 1);
        assert_eq!(listed.data[0].trace_id, "alpha");
        assert!(listed.data[0].agent_names.contains(&agent));
    }
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
#[case::successful(false)]
#[case::failed(true)]
#[tokio::test]
async fn histogram_counts_runs_without_an_agent_or_service(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] failed: bool,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!(T0_MS * 1_000_000)),
            ("TraceId".into(), json!("unlabelled")),
            ("SpanId".into(), json!("root")),
            ("ParentSpanId".into(), json!("")),
            ("SpanName".into(), json!("run")),
            ("ServiceName".into(), json!("")),
            ("ObservationType".into(), json!("chain")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            (
                "StatusCode".into(),
                json!(if failed {
                    "STATUS_CODE_ERROR"
                } else {
                    "STATUS_CODE_OK"
                }),
            ),
        ])],
    )
    .await?;
    let reader = reader();
    let filter = filter(T0_MS, "trace_id:unlabelled");
    let histogram = reader.histogram(&store, &team_a(), &filter, 3).await?;
    let count = reader.count_traces(&store, &team_a(), &filter).await?;
    let bucket = &histogram.buckets[0];
    assert_eq!(count, 1);
    assert_eq!(bucket.total, count);
    assert_eq!(bucket.failed, u64::from(failed));
    assert_eq!(
        bucket.total,
        bucket.failed + bucket.agents.iter().map(|agent| agent.runs).sum::<u64>()
    );
    Ok(())
}

#[rstest]
#[case::agent(CountValue::PrimaryAgent)]
#[case::attribute(CountValue::Attribute("bucket.tag".into()))]
#[tokio::test]
async fn histogram_counts_use_the_displayed_bucket_boundaries(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] value: CountValue,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        (0..10)
            .map(|offset| {
                BTreeMap::from([
                    ("Timestamp".into(), json!((T0_MS + offset) * 1_000_000)),
                    ("TraceId".into(), json!(format!("bucket-{offset}"))),
                    ("SpanId".into(), json!("root")),
                    ("ParentSpanId".into(), json!("")),
                    ("SpanName".into(), json!("run")),
                    ("ServiceName".into(), json!("svc")),
                    ("ObservationType".into(), json!("chain")),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                    ("SpanAttributes".into(), json!({"bucket.tag": "tag"})),
                ])
            })
            .collect(),
    )
    .await?;
    let filter = RunFilter {
        start_ms: T0_MS,
        end_ms: T0_MS + 10,
        search: RunSearch::parse("trace_id:bucket*").unwrap(),
        ..Default::default()
    };
    let counts = store
        .run_counts(
            &team_a(),
            &RunCountQuery {
                filter: filter.clone(),
                by: CountBy {
                    buckets: Some(3),
                    value: Some(value),
                    ..Default::default()
                },
                contains: String::new(),
                limit: None,
            },
        )
        .await?;
    let histogram = litellm_traces::search::histogram(&counts, filter.window(), 3);
    for bucket in histogram.buckets {
        let expected = (T0_MS..T0_MS + 10)
            .filter(|start| (bucket.start_ms..bucket.end_ms).contains(start))
            .count() as u64;
        assert_eq!(bucket.total, expected);
    }
    Ok(())
}

#[rstest]
#[case::agents_in_scope(RunField::Agent, "", &["researcher", "writer"])]
#[case::names_by_frequency(RunField::Name, "", &["plan trip", "write report"])]
#[case::root_statuses_by_frequency(RunField::RootStatus, "", &["ok"])]
#[case::run_errors_by_frequency(RunField::HasError, "", &["false", "true"])]
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
#[case::by_status(RunField::Name, "has_error:true", &["write report"])]
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
                .list_traces(
                    &store,
                    &team_a(),
                    &filter(0, &q),
                    RunOrder::NEWEST,
                    &page(None, 50),
                )
                .await?;
            let excluded = reader
                .list_traces(
                    &store,
                    &team_a(),
                    &filter(0, &format!("-{q}")),
                    RunOrder::NEWEST,
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

async fn sql(fixture: &SeededDatabase, query: String) -> TestResult<String> {
    let writer = Connection::writer(&fixture.database.url)?;
    Ok(fixture
        .database
        .client
        .post(writer.url().clone())
        .body(query)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?
        .trim()
        .to_owned())
}

async fn table_rows(fixture: &SeededDatabase, table: &str) -> TestResult<u64> {
    Ok(
        sql(fixture, format!("SELECT count() FROM {DATABASE}.{table}"))
            .await?
            .parse()?,
    )
}

async fn rows_read_by(fixture: &SeededDatabase, marker: &str) -> TestResult<u64> {
    sql(fixture, "SYSTEM FLUSH LOGS".into()).await?;
    let read = sql(
        fixture,
        format!(
            "SELECT read_rows FROM system.query_log WHERE type = 'QueryFinish' \
             AND current_database = '{DATABASE}' AND position(query, '{marker}') > 0 \
             AND query NOT LIKE '%system.query_log%' \
             ORDER BY event_time_microseconds DESC LIMIT 1"
        ),
    )
    .await?;
    if read.is_empty() {
        return Err(format!("no finished query mentions {marker}").into());
    }
    Ok(read.parse()?)
}

#[rstest]
#[tokio::test]
async fn listed_runs_name_the_agent_they_matched_even_when_resolution_is_limited(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!((T0_MS + HOUR_MS + 5) * 1_000_000)),
            ("TraceId".into(), json!("beta")),
            ("SpanId".into(), json!("beta-oversized")),
            ("ParentSpanId".into(), json!("beta-root")),
            (
                "SpanName".into(),
                json!("x".repeat(litellm_storage_clickhouse::READ_LIMITS.response_bytes + 1)),
            ),
            ("ObservationType".into(), json!("tool")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("Duration".into(), json!(1_000_000)),
        ])],
    )
    .await?;
    let reader = reader();
    let agents = reader
        .values(&store, &team_a(), &filter(0, ""), RunField::Agent, "", 10)
        .await?;
    let mut limited = 0;
    for agent in &agents.values {
        let listed = reader
            .list_traces(
                &store,
                &team_a(),
                &filter(0, &format!(r#"agent:"{agent}""#)),
                RunOrder::NEWEST,
                &page(None, 50),
            )
            .await?;
        assert!(!listed.data.is_empty(), "agent:{agent} lists nothing");
        for run in &listed.data {
            limited += usize::from(run.resolution_limited);
            assert!(
                run.agent_names.contains(agent),
                "{} matched agent:{agent} but lists {:?}",
                run.trace_id,
                run.agent_names
            );
        }
    }
    assert_eq!(
        limited, 1,
        "the oversized span should leave exactly one run in its summary"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn listing_runs_skips_out_of_window_span_rows(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        (0..5000)
            .map(|index| {
                BTreeMap::from([
                    (
                        "Timestamp".into(),
                        json!((T0_MS - 24 * HOUR_MS) * 1_000_000),
                    ),
                    ("TraceId".into(), json!("historical")),
                    ("SpanId".into(), json!(format!("historical-{index}"))),
                    ("ParentSpanId".into(), json!("")),
                    ("SpanName".into(), json!("run")),
                    ("ObservationType".into(), json!("chain")),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                ])
            })
            .collect(),
    )
    .await?;
    let query = RunQuery {
        selection: RunSelection::Matching(filter(T0_MS, "")),
        order: RunOrder::NEWEST,
        after: None,
        limit: 50,
    };
    let listed = store.runs(&QueryScope::All, &query).await?;
    assert_eq!(listed.len(), 4);
    let list_read = rows_read_by(&fixture, "FROM owned_core").await?;
    let reader = reader();
    let id = &listed[0].trace_ref;
    let metadata = reader
        .get_trace_metadata(&store, &QueryScope::All, id)
        .await?
        .unwrap();
    assert_eq!(metadata.summary.trace_ref, *id);
    let spans = reader
        .get_trace_spans(&store, &QueryScope::All, id, None, 500)
        .await?
        .unwrap();
    assert_eq!(spans.data.len() as u64, metadata.summary.span_count);
    let budget = 4 * table_rows(&fixture, "spans_core").await?
        + 3 * runs().iter().flat_map(rows).count() as u64;
    let read = rows_read_by(&fixture, "FROM owned_core").await?;
    for (operation, read) in [("list", list_read), ("canonical ID lookup", read)] {
        assert!(
            read <= budget,
            "{operation} read {read} rows, exceeding the {budget}-row budget for bounded candidate, ownership and canonical scans"
        );
    }
    Ok(())
}

async fn peak_memory_of(fixture: &SeededDatabase, marker: &str) -> TestResult<u64> {
    sql(fixture, "SYSTEM FLUSH LOGS".into()).await?;
    let peak = sql(
        fixture,
        format!(
            "SELECT max(memory_usage) FROM system.query_log WHERE type = 'QueryFinish' \
             AND current_database = '{DATABASE}' AND position(query, '{marker}') > 0 \
             AND query NOT LIKE '%system.query_log%'"
        ),
    )
    .await?;
    Ok(peak.parse()?)
}

#[rstest]
#[tokio::test]
async fn newest_page_aggregates_recent_runs_instead_of_the_whole_window(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let writer = Connection::writer(&fixture.database.url)?;
    let (hours, runs_per_hour) = (24, 500);
    let start_ms = |index: i64| T0_MS + index * HOUR_MS / runs_per_hour;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        (0..hours * runs_per_hour)
            .map(|index| {
                BTreeMap::from([
                    ("Timestamp".into(), json!(start_ms(index) * 1_000_000)),
                    ("TraceId".into(), json!(format!("run-{index:05}"))),
                    ("SpanId".into(), json!("root")),
                    ("ParentSpanId".into(), json!("")),
                    ("SpanName".into(), json!(format!("run-{index}"))),
                    ("Input".into(), json!(format!("{index}{}", "x".repeat(240)))),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                ])
            })
            .collect(),
    )
    .await?;
    let connection = fixture
        .readers
        .connection(&fixture.database.client, &QueryScope::All, "fixture-secret")
        .await?;
    let store = ClickHouseTraces::new(fixture.database.client.clone(), connection);
    let window = RunFilter {
        start_ms: T0_MS,
        end_ms: T0_MS + hours * HOUR_MS,
        ..Default::default()
    };
    let listed = store
        .runs(
            &QueryScope::All,
            &RunQuery {
                selection: RunSelection::Matching(window.clone()),
                order: RunOrder::NEWEST,
                after: None,
                limit: 10,
            },
        )
        .await?;
    let total = hours * runs_per_hour;
    assert_eq!(
        listed.iter().map(|run| run.start_ms).collect::<Vec<_>>(),
        (total - 10..total).rev().map(start_ms).collect::<Vec<_>>()
    );
    let counted = store
        .run_counts(
            &QueryScope::All,
            &RunCountQuery {
                filter: window,
                by: CountBy::default(),
                contains: String::new(),
                limit: None,
            },
        )
        .await?;
    assert_eq!(counted[0].runs, total as u64);
    let page = peak_memory_of(&fixture, "page AS (").await?;
    let whole_window = peak_memory_of(&fixture, "ARRAY JOIN multiIf(").await?;
    assert!(
        page * 4 < whole_window,
        "the newest page used {page} bytes, close to the {whole_window} bytes of aggregating every run in the window"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn reading_the_spans_of_listed_runs_skips_other_runs_in_the_window(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let client = &fixture.database.client;
    let writer = Connection::writer(&fixture.database.url)?;
    let rows_for = |index: i64| -> Vec<BTreeMap<String, Value>> {
        let trace_id = format!("run-{index:02}");
        let start_ms = T0_MS + index * 60_000;
        (0..3)
            .map(|offset| {
                BTreeMap::from([
                    ("Timestamp".into(), json!((start_ms + offset) * 1_000_000)),
                    ("TraceId".into(), json!(trace_id)),
                    ("SpanId".into(), json!(format!("{trace_id}-{offset}"))),
                    (
                        "ParentSpanId".into(),
                        json!(if offset == 0 {
                            String::new()
                        } else {
                            format!("{trace_id}-0")
                        }),
                    ),
                    ("SpanName".into(), json!("step")),
                    ("ServiceName".into(), json!("svc")),
                    (
                        "ObservationType".into(),
                        json!(if offset == 0 { "chain" } else { "tool" }),
                    ),
                    ("TeamId".into(), json!("team-a")),
                    ("ApiKeyHash".into(), json!("key-a")),
                    ("Duration".into(), json!(1_000_000)),
                ])
            })
            .collect()
    };
    for index in 0..12 {
        insert_rows(
            client,
            &writer,
            DATABASE,
            InsertTable::OtelTraces,
            rows_for(index),
        )
        .await?;
    }
    let connection = fixture
        .readers
        .connection(client, &QueryScope::All, "fixture-secret")
        .await?;
    let store = ClickHouseTraces::new(client.clone(), connection);
    let run = store
        .runs(
            &QueryScope::All,
            &RunQuery {
                selection: RunSelection::TraceId("run-05".into()),
                order: RunOrder::NEWEST,
                after: None,
                limit: 2,
            },
        )
        .await?;
    let spans = store
        .spans(
            &QueryScope::All,
            &SpanQuery {
                selection: SpanSelection::Runs {
                    trace_ids: vec![run[0].trace_id.clone()],
                    trace_refs: vec![run[0].trace_ref.clone()],
                    window: T0_MS..WINDOW_END_MS,
                },
                as_of_ms: u64::MAX,
                after: None,
                limit: 256,
            },
        )
        .await?;
    assert_eq!(spans.len(), 3);
    let read = rows_read_by(&fixture, &run[0].trace_ref).await?;
    assert!(
        read <= 2 * spans.len() as u64,
        "reading one run's 3 spans read {read} of the 36 spans in the window"
    );
    Ok(())
}

async fn add_attributes(fixture: &SeededDatabase) -> TestResult {
    let writer = Connection::writer(&fixture.database.url)?;
    let span = |trace_id: &str, start_ms: i64, column: &str, value: &str| {
        BTreeMap::from([
            ("Timestamp".into(), json!((start_ms + 9) * 1_000_000)),
            ("TraceId".into(), json!(trace_id)),
            ("SpanId".into(), json!(format!("{trace_id}-tagged"))),
            ("ParentSpanId".into(), json!(format!("{trace_id}-root"))),
            ("SpanName".into(), json!("tag")),
            ("ServiceName".into(), json!("svc")),
            ("ObservationType".into(), json!("tool")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("Duration".into(), json!(1_000_000)),
            (column.into(), json!({"tenant.tier": value})),
        ])
    };
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![
            span("alpha", T0_MS, "SpanAttributes", "gold"),
            span("beta", T0_MS + HOUR_MS, "ResourceAttributes", "silver"),
        ],
    )
    .await?;
    Ok(())
}

#[rstest]
#[case::service("service:svc", &["gamma", "beta", "alpha"])]
#[case::other_service("service:other", &[])]
#[case::team("team:team-a", &["gamma", "beta", "alpha"])]
#[case::excluded_team("-team:team-a", &[])]
#[case::span_attribute("attr.tenant.tier:gold", &["alpha"])]
#[case::resource_attribute("attr.tenant.tier:SILV*", &["beta"])]
#[case::excluded_attribute("-attr.tenant.tier:gold", &["gamma", "beta"])]
#[case::any_attribute("attr.tenant.tier:*", &["beta", "alpha"])]
#[case::missing_attribute("-attr.tenant.tier:*", &["gamma"])]
#[case::missing_wildcard("attr.missing:*", &[])]
#[case::two_attributes("attr.tenant.tier:gold attr.tenant.tier:silver", &[])]
#[case::attribute_and_field("attr.tenant.tier:* has_error:true", &["beta"])]
#[case::unknown_attribute("attr.missing:gold", &[])]
#[tokio::test]
async fn service_team_and_attribute_filters_select_runs(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] q: &str,
    #[case] expected: &[&str],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    add_attributes(&fixture).await?;
    let page = reader()
        .list_traces(
            &store,
            &team_a(),
            &filter(0, q),
            RunOrder::NEWEST,
            &page(None, 50),
        )
        .await?;
    let listed: Vec<&str> = page.data.iter().map(|run| run.trace_id.as_str()).collect();
    assert_eq!(listed, expected);
    Ok(())
}

#[rstest]
#[case::duration_ascending(RunSortKey::DurationMs, false)]
#[case::duration_descending(RunSortKey::DurationMs, true)]
#[case::span_count_ascending(RunSortKey::SpanCount, false)]
#[case::span_count_descending(RunSortKey::SpanCount, true)]
#[case::error_count_ascending(RunSortKey::ErrorCount, false)]
#[case::error_count_descending(RunSortKey::ErrorCount, true)]
#[case::start_ascending(RunSortKey::StartMs, false)]
#[case::start_descending(RunSortKey::StartMs, true)]
#[case::reference_ascending(RunSortKey::TraceRef, false)]
#[case::reference_descending(RunSortKey::TraceRef, true)]
#[tokio::test]
async fn metric_sorting_pages_by_the_deduplicated_displayed_values(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] key: RunSortKey,
    #[case] descending: bool,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let writer = Connection::writer(&fixture.database.url)?;
    let span = |trace: &str, id: &str, duration: u64, failed: bool| {
        BTreeMap::from([
            ("Timestamp".into(), json!(T0_MS * 1_000_000)),
            ("TraceId".into(), json!(trace)),
            ("SpanId".into(), json!(id)),
            (
                "ParentSpanId".into(),
                json!(if id == "root" { "" } else { "root" }),
            ),
            ("SpanName".into(), json!(id)),
            ("ServiceName".into(), json!("svc")),
            (
                "ObservationType".into(),
                json!(if id == "root" { "chain" } else { "tool" }),
            ),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("Duration".into(), json!(duration)),
            ("EngineReceivedMs".into(), json!(1)),
            (
                "StatusCode".into(),
                json!(if failed {
                    "STATUS_CODE_ERROR"
                } else {
                    "STATUS_CODE_OK"
                }),
            ),
        ])
    };
    let duplicate = |changes: [(&str, serde_json::Value); 1]| {
        BTreeMap::from_iter(
            span("metric-a", "root", 50_000_000, true)
                .into_iter()
                .chain(changes.into_iter().map(|(key, value)| (key.into(), value))),
        )
    };
    fixture
        .database
        .client
        .post(writer.url().clone())
        .query(&[(
            "query",
            format!("INSERT INTO {DATABASE}.otel_traces FORMAT JSONEachRow"),
        )])
        .body(encode_rows(vec![
            span("metric-a", "root", 400_000, false),
            span("metric-b", "root", 900_000, false),
            span("metric-b", "child", 200_000, false),
            span("metric-c", "root", 1_100_000, false),
            span("metric-c", "child-one", 200_000, true),
            span("metric-c", "child-two", 300_000, true),
            duplicate([("EngineReceivedMs", json!(2))]),
            duplicate([("Timestamp", json!((T0_MS + 1) * 1_000_000))]),
            duplicate([("StatusMessage", json!("later"))]),
            span("metric-b", "root", 800_000, false),
            span("metric-b", "root", 800_000, true),
        ])?)
        .send()
        .await?
        .error_for_status()?;
    let reader = reader();
    let order = RunOrder { key, descending };
    let filter = filter(T0_MS, "trace_id:metric*");
    let mut listed = Vec::new();
    let mut cursor = None;
    loop {
        let result = reader
            .list_traces(&store, &team_a(), &filter, order, &page(cursor.take(), 1))
            .await?;
        listed.extend(result.data);
        let Some(next) = result.next_cursor else {
            break;
        };
        cursor = Some(next);
    }
    let expected = if descending {
        ["metric-c", "metric-b", "metric-a"]
    } else {
        ["metric-a", "metric-b", "metric-c"]
    };
    if matches!(key, RunSortKey::StartMs | RunSortKey::TraceRef) {
        assert_eq!(
            listed
                .iter()
                .map(|run| run.trace_id.as_str())
                .collect::<BTreeSet<_>>(),
            expected.into_iter().collect()
        );
    } else {
        assert_eq!(
            listed
                .iter()
                .map(|run| run.trace_id.as_str())
                .collect::<Vec<_>>(),
            expected
        );
    }
    for run in listed {
        let (duration_ns, span_count, error_count) = match run.trace_id.as_str() {
            "metric-a" => (400_000, 1, 0),
            "metric-b" => (800_000, 2, 1),
            "metric-c" => (1_100_000, 3, 2),
            other => return Err(format!("unexpected run {other}").into()),
        };
        assert_eq!(run.duration_ms, f64::from(duration_ns) / 1_000_000.0);
        assert_eq!(run.span_count, span_count);
        assert_eq!(run.error_count, error_count);
    }
    let errors = RunFilter {
        search: RunSearch::parse("trace_id:metric* has_error:true").unwrap(),
        ..filter.clone()
    };
    let filtered = reader
        .list_traces(&store, &team_a(), &errors, order, &page(None, 10))
        .await?;
    assert_eq!(
        filtered
            .data
            .iter()
            .map(|run| run.trace_id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["metric-b", "metric-c"])
    );
    let count = reader.count_traces(&store, &team_a(), &errors).await?;
    let histogram = reader.histogram(&store, &team_a(), &errors, 3).await?;
    assert_eq!(count, filtered.data.len() as u64);
    assert_eq!(
        histogram
            .buckets
            .iter()
            .map(|bucket| bucket.total)
            .sum::<u64>(),
        count
    );
    assert_eq!(
        histogram
            .buckets
            .iter()
            .map(|bucket| bucket.failed)
            .sum::<u64>(),
        count
    );
    let all = reader.histogram(&store, &team_a(), &filter, 3).await?;
    assert_eq!(
        all.buckets.iter().map(|bucket| bucket.failed).sum::<u64>(),
        count
    );
    Ok(())
}

#[rstest]
#[case::newest(RunOrder::NEWEST)]
#[case::oldest(RunOrder { descending: false, ..RunOrder::NEWEST })]
#[case::longest(RunOrder { key: RunSortKey::DurationMs, descending: true })]
#[case::shortest(RunOrder { key: RunSortKey::DurationMs, descending: false })]
#[case::most_spans(RunOrder { key: RunSortKey::SpanCount, descending: true })]
#[case::fewest_spans(RunOrder { key: RunSortKey::SpanCount, descending: false })]
#[case::most_errors(RunOrder { key: RunSortKey::ErrorCount, descending: true })]
#[case::fewest_errors(RunOrder { key: RunSortKey::ErrorCount, descending: false })]
#[case::by_reference(RunOrder::BY_REFERENCE)]
#[case::by_reference_descending(RunOrder { descending: true, ..RunOrder::BY_REFERENCE })]
#[tokio::test]
async fn one_run_pages_walk_every_order_without_gaps_or_repeats(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] order: RunOrder,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let reader = reader();
    let all = RunQuery {
        selection: RunSelection::Matching(filter(0, "")),
        order: RunOrder::NEWEST,
        after: None,
        limit: 50,
    };
    let mut expected = store.runs(&team_a(), &all).await?;
    expected.sort_by(|left, right| order.compare(left, right));
    let expected: Vec<String> = expected.into_iter().map(|run| run.trace_id).collect();

    let mut listed = Vec::new();
    let mut cursor = None;
    loop {
        let request = PageRequest {
            cursor: cursor.take(),
            limit: 1,
        };
        let page = reader
            .list_traces(&store, &team_a(), &filter(0, ""), order, &request)
            .await?;
        listed.extend(page.data.into_iter().map(|run| run.trace_id));
        let Some(next) = page.next_cursor else { break };
        cursor = Some(next);
    }
    assert_eq!(listed, expected);

    let whole = reader
        .list_traces(
            &store,
            &team_a(),
            &filter(0, ""),
            order,
            &PageRequest {
                cursor: None,
                limit: 3,
            },
        )
        .await?;
    assert_eq!(whole.data.len(), 3);
    assert!(whole.next_cursor.is_none());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn listed_references_restrict_runs_and_order_by_reference_pages_stably(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let reader = reader();
    let by_reference = |cursor| PageRequest { cursor, limit: 2 };
    let first = reader
        .list_traces(
            &store,
            &team_a(),
            &filter(0, ""),
            RunOrder::BY_REFERENCE,
            &by_reference(None),
        )
        .await?;
    let second = reader
        .list_traces(
            &store,
            &team_a(),
            &filter(0, ""),
            RunOrder::BY_REFERENCE,
            &by_reference(first.next_cursor.clone()),
        )
        .await?;
    let refs: Vec<String> = first
        .data
        .iter()
        .chain(&second.data)
        .map(|run| run.trace_ref.clone())
        .collect();
    let mut sorted = refs.clone();
    sorted.sort();
    assert_eq!(refs, sorted);
    assert_eq!(refs.len(), 3);
    assert!(second.next_cursor.is_none());

    let picked = RunFilter {
        trace_refs: vec![refs[1].clone(), "not-a-run".into()],
        ..filter(0, "")
    };
    let only = reader
        .list_traces(
            &store,
            &team_a(),
            &picked,
            RunOrder::NEWEST,
            &page(None, 50),
        )
        .await?;
    assert_eq!(
        only.data
            .iter()
            .map(|run| &run.trace_ref)
            .collect::<Vec<_>>(),
        [&refs[1]]
    );
    assert_eq!(reader.count_traces(&store, &team_a(), &picked).await?, 1);
    assert_eq!(
        reader
            .count_traces(&store, &team_a(), &filter(0, "model:gpt-x"))
            .await?,
        2
    );
    Ok(())
}

#[rstest]
#[case::keys(CountValue::AttributeKey, "", &[("tenant.tier", 2)])]
#[case::values(CountValue::Attribute("tenant.tier".into()), "", &[("gold", 1), ("silver", 1)])]
#[case::values_containing(CountValue::Attribute("tenant.tier".into()), "IL", &[("silver", 1)])]
#[tokio::test]
async fn attribute_keys_and_values_count_runs_in_scope(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] value: CountValue,
    #[case] contains: &str,
    #[case] expected: &[(&str, u64)],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    add_attributes(&fixture).await?;
    let query = RunCountQuery {
        filter: filter(0, ""),
        by: CountBy {
            value: Some(value),
            ..CountBy::default()
        },
        contains: contains.into(),
        limit: Some(10),
    };
    let counts = store.run_counts(&team_a(), &query).await?;
    let counted: Vec<(&str, u64)> = counts
        .iter()
        .map(|count| (count.value.as_str(), count.runs))
        .collect();
    assert_eq!(counted, expected);
    Ok(())
}

#[rstest]
#[case::whole(TextRange::ALL, None, &[("alpha-root", "book a flight to Paris", false)])]
#[case::window(TextRange::From { offset: 5, max_chars: Some(6) }, None, &[("alpha-root", "a flig", false)])]
#[case::tail(TextRange::Last { chars: 5 }, None, &[("alpha-root", "Paris", false)])]
#[case::tail_longer_than_text(TextRange::Last { chars: 500 }, None, &[("alpha-root", "book a flight to Paris", false)])]
#[case::contains(TextRange::From { offset: 0, max_chars: Some(0) }, Some("flight to"), &[("alpha-root", "", true)])]
#[case::contains_is_case_sensitive(TextRange::From { offset: 0, max_chars: Some(0) }, Some("PARIS"), &[("alpha-root", "", false)])]
#[tokio::test]
async fn span_text_reads_ranges_of_each_listed_span(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    #[case] range: TextRange,
    #[case] contains: Option<&str>,
    #[case] expected: &[(&str, &str, bool)],
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let runs = store
        .runs(
            &team_a(),
            &RunQuery {
                selection: RunSelection::TraceId("alpha".into()),
                order: RunOrder::NEWEST,
                after: None,
                limit: 2,
            },
        )
        .await?;
    let texts = reader()
        .span_text(
            &store,
            &team_a(),
            "alpha",
            &runs[0].trace_ref,
            vec!["alpha-root".into(), "alpha-llm".into(), "missing".into()],
            SpanPart::Input,
            range,
            contains.map(str::to_owned),
        )
        .await?;
    let inputs: Vec<(&str, &str, bool)> = texts
        .iter()
        .filter(|text| text.total_chars > 0)
        .map(|text| (text.span_id.as_str(), text.text.as_str(), text.contains))
        .collect();
    assert_eq!(inputs, expected);
    assert_eq!(
        texts
            .iter()
            .map(|text| text.span_id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["alpha-llm", "alpha-root"])
    );
    let foreign = reader()
        .span_text(
            &store,
            &QueryScope::Owned {
                user_id: String::new(),
                team_ids: vec!["team-b".into()],
            },
            "alpha",
            &runs[0].trace_ref,
            vec!["alpha-root".into()],
            SpanPart::Input,
            range,
            None,
        )
        .await?;
    assert!(foreign.is_empty());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn list_snapshot_excludes_late_spans_and_preserves_sorted_pages(
    #[future] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database.await?;
    let writer = Connection::writer(&fixture.database.url)?;
    let store = ClickHouseTraces::new(
        fixture.database.client.clone(),
        Connection::configured(&fixture.database.url, DATABASE, "default", "")?,
    );
    let row = |trace: &str, span: &str, duration: u64, received: u64, error: bool, offset: i64| {
        BTreeMap::from([
            ("Timestamp".into(), json!((T0_MS + offset) * 1_000_000)),
            ("TraceId".into(), json!(trace)),
            ("SpanId".into(), json!(span)),
            (
                "ParentSpanId".into(),
                json!(if span == "root" { "" } else { "root" }),
            ),
            ("SpanName".into(), json!(span)),
            ("ServiceName".into(), json!("snapshot")),
            ("ObservationType".into(), json!("chain")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("Duration".into(), json!(duration)),
            ("EngineReceivedMs".into(), json!(received)),
            (
                "StatusCode".into(),
                json!(if error {
                    "STATUS_CODE_ERROR"
                } else {
                    "STATUS_CODE_OK"
                }),
            ),
        ])
    };
    let insert = |rows: Vec<BTreeMap<String, Value>>| {
        let client = fixture.database.client.clone();
        let url = writer.url().clone();
        async move {
            client
                .post(url)
                .query(&[(
                    "query",
                    format!("INSERT INTO {DATABASE}.otel_traces FORMAT JSONEachRow"),
                )])
                .body(encode_rows(rows)?)
                .send()
                .await?
                .error_for_status()?;
            TestResult::Ok(())
        }
    };
    insert(vec![
        row("snapshot-a", "root", 1_000_000, 10, false, 0),
        row("snapshot-b", "root", 2_000_000, 10, false, 0),
        row("snapshot-c", "root", 3_000_000, 10, false, 0),
    ])
    .await?;
    let reader = reader();
    let filter = RunFilter {
        as_of_ms: 10,
        ..filter(T0_MS, "trace_id:snapshot*")
    };
    let order = RunOrder {
        key: RunSortKey::DurationMs,
        descending: false,
    };
    let first = reader
        .list_traces(&store, &team_a(), &filter, order, &page(None, 1))
        .await?;
    assert_eq!(first.data[0].trace_id, "snapshot-a");
    assert_eq!(first.window, filter.window());
    insert(vec![
        row("snapshot-a", "late", 10_000_000, 20, true, 0),
        row("snapshot-new", "root", 100_000, 20, false, 0),
        row("snapshot-c", "root", 100_000_000, 20, true, -1),
        row("snapshot-c", "backdated-child", 1_000_000, 20, true, -1),
    ])
    .await?;
    let live_filter = RunFilter {
        as_of_ms: 20,
        ..filter.clone()
    };
    let live = reader
        .list_traces(&store, &team_a(), &live_filter, order, &page(None, 10))
        .await?;
    let live_a = live
        .data
        .iter()
        .find(|run| run.trace_id == "snapshot-a")
        .unwrap();
    assert!(live_a.has_error);
    assert_eq!(live_a.status, litellm_traces::SpanStatus::Ok);
    let metadata = reader
        .get_trace_metadata(&store, &team_a(), &live_a.trace_ref)
        .await?
        .unwrap();
    assert_eq!(metadata.summary.trace_ref, live_a.trace_ref);
    assert!(metadata.summary.has_error);
    let spans = reader
        .get_trace_spans(&store, &team_a(), &live_a.trace_ref, None, 1)
        .await?
        .unwrap();
    let next = reader
        .get_trace_spans(
            &store,
            &team_a(),
            &live_a.trace_ref,
            spans.next_cursor.as_deref(),
            1,
        )
        .await?
        .unwrap();
    assert_ne!(spans.data[0].span_id, next.data[0].span_id);
    assert!(next.next_cursor.is_none());

    let second = reader
        .list_traces(
            &store,
            &team_a(),
            &filter,
            order,
            &page(first.next_cursor, 1),
        )
        .await?;
    let third = reader
        .list_traces(
            &store,
            &team_a(),
            &filter,
            order,
            &page(second.next_cursor, 1),
        )
        .await?;
    assert_eq!(second.data[0].trace_id, "snapshot-b");
    assert_eq!(third.data[0].trace_id, "snapshot-c");
    assert_eq!(third.data[0].duration_ms, 3.0);
    assert!(!third.data[0].has_error);
    assert!(third.next_cursor.is_none());
    let repeated = reader
        .list_traces(&store, &team_a(), &filter, order, &page(None, 10))
        .await?;
    assert_eq!(repeated.data[0].duration_ms, 1.0);
    assert!(!repeated.data[0].has_error);
    assert_eq!(reader.count_traces(&store, &team_a(), &filter).await?, 3);
    let histogram = reader.histogram(&store, &team_a(), &filter, 3).await?;
    assert_eq!(histogram.window, filter.window());
    assert_eq!(
        histogram
            .buckets
            .iter()
            .map(|bucket| bucket.total)
            .sum::<u64>(),
        3
    );
    let errors = RunFilter {
        search: RunSearch::parse("has_error:true root_status:ok").unwrap(),
        ..live_filter
    };
    let errors = reader
        .list_traces(&store, &team_a(), &errors, order, &page(None, 10))
        .await?;
    assert_eq!(errors.data.len(), 1);
    assert_eq!(errors.data[0].trace_id, "snapshot-a");
    Ok(())
}
