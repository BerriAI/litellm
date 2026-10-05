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
        ..Default::default()
    }
}

fn page(cursor: Option<String>, limit: u32) -> PageRequest {
    PageRequest {
        cursor,
        limit,
        ..Default::default()
    }
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
        "the oversized span should leave exactly one run on its rollup summary"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn listing_runs_scans_the_rollup_once(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    let store = seed(&fixture).await?;
    let query = RunQuery {
        selection: RunSelection::Matching(filter(0, "")),
        order: RunOrder::NEWEST,
        after: None,
        limit: 50,
    };
    let listed = store.runs(&QueryScope::All, &query).await?;
    assert_eq!(listed.len(), 4);
    let budget = table_rows(&fixture, "agent_traces_by_key").await?
        + table_rows(&fixture, "otel_traces").await?;
    let read = rows_read_by(&fixture, "FROM owned_runs").await?;
    assert!(
        read <= budget,
        "listing 4 runs read {read} rows, more than the {budget} rollup and span rows that exist"
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
#[case::two_attributes("attr.tenant.tier:gold attr.tenant.tier:silver", &[])]
#[case::attribute_and_field("attr.tenant.tier:* status:error", &["beta"])]
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
