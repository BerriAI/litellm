use std::collections::BTreeMap;

use litellm_traces_clickhouse::{
    Connection, InsertTable, Parameter, ReadQuery, ensure_schema, execute_named_read, execute_read,
    insert_rows,
};
use rstest::rstest;
use serde_json::{Value, json};
use time::{Duration, OffsetDateTime, format_description::well_known::Rfc3339};

mod support;

use support::{ClickHouseDatabase, TestResult, database};

struct Feedback<'a> {
    team: &'a str,
    key: &'a str,
    trace: &'a str,
    author: &'a str,
    score: u8,
    comment: &'a str,
    edited_after_seconds: i64,
    deleted: bool,
}

const SAVED: Feedback<'static> = Feedback {
    team: "team-a",
    key: "key-a",
    trace: "trace-1",
    author: "alice",
    score: 3,
    comment: "missed the file",
    edited_after_seconds: 0,
    deleted: false,
};

fn now() -> TestResult<OffsetDateTime> {
    Ok(OffsetDateTime::now_utc().replace_millisecond(0)?)
}

fn iso(at: OffsetDateTime) -> TestResult<String> {
    Ok(at.format(&Rfc3339)?)
}

fn row(feedback: &Feedback, created: OffsetDateTime) -> TestResult<BTreeMap<String, Value>> {
    let updated = created + Duration::seconds(feedback.edited_after_seconds);
    Ok(serde_json::from_value(json!({
        "TeamId": feedback.team, "ApiKeyHash": feedback.key, "TraceId": feedback.trace,
        "Author": feedback.author, "Score": feedback.score, "Comment": feedback.comment,
        "CreatedAt": iso(created)?, "UpdatedAt": iso(updated)?,
        "IsDeleted": u8::from(feedback.deleted)
    }))?)
}

async fn ready(database: &ClickHouseDatabase) -> TestResult<Connection> {
    let writer = Connection::writer(&database.url)?;
    ensure_schema(&database.client, &writer, "trace_test", 7).await?;
    Ok(writer)
}

async fn save(
    database: &ClickHouseDatabase,
    writer: &Connection,
    created: OffsetDateTime,
    feedback: &[Feedback<'_>],
) -> TestResult {
    let rows = feedback
        .iter()
        .map(|entry| row(entry, created))
        .collect::<TestResult<Vec<_>>>()?;
    insert_rows(
        &database.client,
        writer,
        "trace_test",
        InsertTable::LensFeedback,
        rows,
    )
    .await?;
    Ok(())
}

fn access(team: &str) -> BTreeMap<String, Parameter> {
    BTreeMap::from([
        (
            "all_teams".into(),
            Parameter::Integer(i64::from(team.is_empty())),
        ),
        ("team".into(), Parameter::Text(team.into())),
        ("key_hash".into(), Parameter::Text(String::new())),
    ])
}

async fn read(
    database: &ClickHouseDatabase,
    query: ReadQuery,
    parameters: BTreeMap<String, Parameter>,
) -> TestResult<Vec<Value>> {
    let reader = Connection::configured(&database.url, "trace_test", "default", "")?;
    let body: Value = serde_json::from_str(
        &execute_named_read(&database.client, &reader, query, &parameters).await?,
    )?;
    Ok(body["data"]
        .as_array()
        .cloned()
        .ok_or_else(|| format!("no data in {body}"))?)
}

async fn trace_ref(
    database: &ClickHouseDatabase,
    team: &str,
    key: &str,
    trace: &str,
) -> TestResult<String> {
    let reader = Connection::configured(&database.url, "trace_test", "default", "")?;
    let sql = format!(
        "SELECT hex(SHA256(concat('{team}', char(0), '{key}', char(0), '{trace}'))) AS trace_ref"
    );
    let response: Value = serde_json::from_str(
        &execute_read(&database.client, &reader, &sql, &BTreeMap::new()).await?,
    )?;
    Ok(response["data"][0]["trace_ref"]
        .as_str()
        .unwrap_or_default()
        .to_owned())
}

async fn feedback(
    database: &ClickHouseDatabase,
    team: &str,
    trace: &str,
    trace_ref: &str,
) -> TestResult<Vec<Value>> {
    let mut parameters = access(team);
    parameters.insert("trace_id".into(), Parameter::Text(trace.into()));
    parameters.insert("trace_ref".into(), Parameter::Text(trace_ref.into()));
    read(database, ReadQuery::Feedback, parameters).await
}

fn timestamp(row: &Value, field: &str) -> TestResult<OffsetDateTime> {
    Ok(OffsetDateTime::parse(
        row[field].as_str().unwrap_or_default(),
        &Rfc3339,
    )?)
}

#[rstest]
#[tokio::test]
async fn a_later_save_replaces_the_authors_feedback_and_keeps_other_authors(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;
    let created = now()?;
    save(&database, &writer, created, &[SAVED]).await?;
    save(
        &database,
        &writer,
        created,
        &[
            Feedback {
                score: 8,
                comment: "fine after retry",
                edited_after_seconds: 300,
                ..SAVED
            },
            Feedback {
                author: "bob",
                score: 10,
                comment: "",
                ..SAVED
            },
        ],
    )
    .await?;
    let reference = trace_ref(&database, "team-a", "key-a", "trace-1").await?;

    let rows = feedback(&database, "", "trace-1", &reference).await?;

    let shown: Vec<(&str, u64, &str)> = rows
        .iter()
        .map(|row| {
            (
                row["author"].as_str().unwrap_or_default(),
                row["score"].as_u64().unwrap_or(99),
                row["comment"].as_str().unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(shown, [("alice", 8, "fine after retry"), ("bob", 10, "")]);
    assert_eq!(timestamp(&rows[0], "created_at")?, created);
    assert_eq!(
        timestamp(&rows[0], "updated_at")?,
        created + Duration::seconds(300)
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn a_deleted_version_hides_the_authors_feedback(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;
    let created = now()?;
    save(
        &database,
        &writer,
        created,
        &[
            SAVED,
            Feedback {
                author: "bob",
                score: 6,
                ..SAVED
            },
        ],
    )
    .await?;
    save(
        &database,
        &writer,
        created,
        &[Feedback {
            deleted: true,
            edited_after_seconds: 540,
            ..SAVED
        }],
    )
    .await?;
    let reference = trace_ref(&database, "team-a", "key-a", "trace-1").await?;

    let authors: Vec<Value> = feedback(&database, "", "trace-1", &reference)
        .await?
        .into_iter()
        .map(|row| row["author"].clone())
        .collect();

    assert_eq!(authors, [json!("bob")]);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn summaries_aggregate_live_feedback_per_trace_and_respect_team_scope(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;
    save(
        &database,
        &writer,
        now()?,
        &[
            Feedback { score: 2, ..SAVED },
            Feedback {
                author: "bob",
                score: 7,
                ..SAVED
            },
            Feedback {
                author: "carol",
                score: 0,
                deleted: true,
                ..SAVED
            },
            Feedback {
                team: "team-b",
                key: "key-b",
                trace: "trace-2",
                score: 9,
                ..SAVED
            },
        ],
    )
    .await?;
    let summary = |team: &str| {
        let mut parameters = access(team);
        parameters.insert(
            "trace_ids".into(),
            Parameter::Strings(vec![
                "trace-1".into(),
                "trace-2".into(),
                "trace-unrated".into(),
            ]),
        );
        parameters
    };

    let everyone = read(&database, ReadQuery::FeedbackSummary, summary("")).await?;
    let team_a = read(&database, ReadQuery::FeedbackSummary, summary("team-a")).await?;

    let by_trace: BTreeMap<&str, (u64, f64, u64)> = everyone
        .iter()
        .map(|row| {
            (
                row["trace_id"].as_str().unwrap_or_default(),
                (
                    row["count"].as_u64().unwrap_or(0),
                    row["average"].as_f64().unwrap_or(-1.0),
                    row["lowest"].as_u64().unwrap_or(99),
                ),
            )
        })
        .collect();
    assert_eq!(
        by_trace,
        BTreeMap::from([("trace-1", (2, 4.5, 2)), ("trace-2", (1, 9.0, 9))])
    );
    assert_eq!(
        team_a
            .iter()
            .map(|row| row["trace_id"].clone())
            .collect::<Vec<_>>(),
        [json!("trace-1")]
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn feedback_for_another_teams_trace_is_not_readable(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;
    save(&database, &writer, now()?, &[SAVED]).await?;
    let reference = trace_ref(&database, "team-a", "key-a", "trace-1").await?;

    assert!(
        feedback(&database, "team-b", "trace-1", &reference)
            .await?
            .is_empty()
    );
    assert_eq!(
        feedback(&database, "team-a", "trace-1", &reference)
            .await?
            .len(),
        1
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn the_table_rejects_scores_above_ten(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;

    let rejected = save(
        &database,
        &writer,
        now()?,
        &[Feedback { score: 11, ..SAVED }],
    )
    .await;

    assert!(rejected.is_err());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn target_resolves_team_and_key_for_a_visible_trace_only(
    #[future(awt)] database: TestResult<ClickHouseDatabase>,
) -> TestResult {
    let database = database?;
    let writer = ready(&database).await?;
    let span: BTreeMap<String, Value> = serde_json::from_value(json!({
        "Timestamp": now()?.unix_timestamp_nanos() as i64, "TraceId": "trace-1", "SpanId": "root",
        "ParentSpanId": "", "ServiceName": "agent", "SpanName": "run",
        "ResourceAttributes": {"litellm.team_id": "team-a", "litellm.api_key_hash": "key-a"}
    }))?;
    insert_rows(
        &database.client,
        &writer,
        "trace_test",
        InsertTable::OtelTraces,
        vec![span],
    )
    .await?;
    let target = |team: &str| {
        let mut parameters = access(team);
        parameters.insert("trace_id".into(), Parameter::Text("trace-1".into()));
        parameters.insert("trace_ref".into(), Parameter::Text(String::new()));
        parameters
    };

    let visible = read(&database, ReadQuery::FeedbackTarget, target("team-a")).await?;
    let hidden = read(&database, ReadQuery::FeedbackTarget, target("team-b")).await?;

    assert_eq!(visible.len(), 1);
    assert_eq!(
        (
            visible[0]["team_id"].as_str(),
            visible[0]["key_hash"].as_str()
        ),
        (Some("team-a"), Some("key-a"))
    );
    assert_eq!(
        visible[0]["trace_ref"].as_str().map(str::to_owned),
        Some(trace_ref(&database, "team-a", "key-a", "trace-1").await?)
    );
    assert!(hidden.is_empty());
    Ok(())
}
