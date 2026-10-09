use std::collections::BTreeMap;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_traces::query::named as contracts;
use litellm_traces_cache::TraceReader;
use litellm_traces_clickhouse::{
    ClickHouseTraces, Connection, InsertTable, QueryScope, encode_rows, execute_read, insert_rows,
    project_spend_rows, query::named::SpendByResponseIds,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database};
use support::TestResult;

const START_MS: i64 = 1_790_000_000_000;

#[fixture]
fn cost_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("request_id".into(), json!("billing-record")),
        ("litellm_call_id".into(), json!("gateway-call")),
        ("response_id".into(), json!("provider-response")),
        ("provider_request_id".into(), json!("provider-request")),
        ("trace_id".into(), json!("transport-trace")),
        ("span_id".into(), json!("model-span")),
        ("team_id".into(), json!("team-a")),
        ("api_key".into(), json!("key-a")),
        ("user".into(), json!("user-a")),
        ("start_time".into(), json!(START_MS)),
        ("end_time".into(), json!(START_MS + 1)),
        ("spend".into(), json!(0.125)),
    ])
}

fn access() -> contracts::ReadAccessParams {
    contracts::ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team-a".into()],
    }
}

async fn execute(fixture: &SeededDatabase, sql: &str) -> TestResult<String> {
    Ok(fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(sql.to_owned())
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?)
}

async fn deliver(fixture: &SeededDatabase, rows: Vec<BTreeMap<String, Value>>) -> TestResult {
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::SpendLogs,
        rows,
    )
    .await?;
    Ok(())
}

async fn lookup(
    fixture: &SeededDatabase,
    connection: &Connection,
    access: contracts::ReadAccessParams,
    id: &str,
) -> TestResult<Vec<contracts::SpendByResponseIdsRow>> {
    let params = contracts::SpendByResponseIdsParams {
        access,
        response_ids: vec![id.into()],
        provider_request_ids: vec![id.into()],
        request_ids: vec![id.into()],
        trace_ids: vec![id.into()],
        start_ms: START_MS - 1,
        end_ms: START_MS + 2,
    };
    Ok(litellm_storage_clickhouse::fetch::<SpendByResponseIds>(
        &fixture.database.client,
        connection,
        &params.into(),
    )
    .await?
    .into_iter()
    .map(|row| row.0)
    .collect())
}

#[rstest]
#[case::cost(Some(0.125), false)]
#[case::zero(Some(0.0), false)]
#[case::unknown(None, false)]
#[case::absent(None, true)]
#[tokio::test]
async fn trace_costs_work_with_raw_select_revoked_in_either_arrival_order(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    mut cost_row: BTreeMap<String, Value>,
    #[case] amount: Option<f64>,
    #[case] absent: bool,
    #[values(false, true)] spend_first: bool,
) -> TestResult {
    let fixture = migrated_database?;
    if absent {
        cost_row.remove("spend");
    } else {
        cost_row.insert("spend".into(), json!(amount));
    }
    if spend_first {
        deliver(&fixture, vec![cost_row.clone()]).await?;
    }
    let writer = Connection::writer(&fixture.database.url)?;
    insert_rows(
        &fixture.database.client,
        &writer,
        DATABASE,
        InsertTable::OtelTraces,
        vec![BTreeMap::from([
            ("Timestamp".into(), json!(START_MS * 1_000_000)),
            ("TraceId".into(), json!("transport-trace")),
            ("SpanId".into(), json!("model-span")),
            ("ObservationType".into(), json!("llm")),
            ("TeamId".into(), json!("team-a")),
            ("ApiKeyHash".into(), json!("key-a")),
            ("UserId".into(), json!("user-a")),
            ("Duration".into(), json!(1_000_000)),
            (
                "CallKeys".into(),
                json!([
                    "litellm_request:gateway-call",
                    "provider_response:provider-response"
                ]),
            ),
            ("CallEvidence".into(), json!("complete")),
        ])],
    )
    .await?;
    execute(&fixture, "CREATE USER native_reader").await?;
    execute(&fixture, "GRANT SELECT ON trace_test.* TO native_reader").await?;
    let connection = Connection::configured(&fixture.database.url, DATABASE, "native_reader", "")?;
    execute(
        &fixture,
        &format!(
            "REVOKE SELECT ON {DATABASE}.spend_logs FROM {}",
            connection.url().username()
        ),
    )
    .await?;
    assert!(
        execute_read(
            &fixture.database.client,
            &connection,
            "SELECT count() FROM spend_logs",
            &BTreeMap::new()
        )
        .await
        .is_err()
    );
    let store = ClickHouseTraces::new(fixture.database.client.clone(), connection);
    let reader = TraceReader::new(litellm_storage_clickhouse::READ_LIMITS.response_bytes);
    if !spend_first {
        let page = reader
            .list_traces(&store, &access(), START_MS - 1, START_MS + 2, None, 50)
            .await?;
        assert_eq!(page.data[0].spend, None);
        deliver(&fixture, vec![cost_row.clone()]).await?;
        tokio::time::sleep(std::time::Duration::from_secs(6)).await;
    }
    deliver(&fixture, vec![cost_row.clone(), cost_row]).await?;
    let page = reader
        .list_traces(&store, &access(), START_MS - 1, START_MS + 2, None, 50)
        .await?;
    assert_eq!(page.data.len(), 1);
    let summary = &page.data[0];
    assert_eq!(
        (summary.spend, summary.priced_calls),
        (amount, u64::from(amount.is_some()))
    );
    let detail = reader
        .get_trace(&store, &access(), &summary.trace_id, &summary.trace_ref)
        .await?
        .ok_or("missing trace")?;
    assert_eq!(detail.summary.spend, amount);
    Ok(())
}

#[rstest]
#[case::deployed_default(false, 3)]
#[case::explicit_deduplication(true, 1)]
#[tokio::test]
async fn exact_retry_repairs_projection_after_raw_insert_succeeds(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    cost_row: BTreeMap<String, Value>,
    #[case] enable_deduplication: bool,
    #[case] expected_raw_rows: usize,
) -> TestResult {
    let fixture = migrated_database?;
    if enable_deduplication {
        execute(&fixture, "ALTER TABLE trace_test.spend_logs MODIFY SETTING non_replicated_deduplication_window = 1000").await?;
    }
    execute(
        &fixture,
        "RENAME TABLE trace_test.lens_call_costs TO trace_test.unavailable_costs",
    )
    .await?;
    assert!(deliver(&fixture, vec![cost_row.clone()]).await.is_err());
    assert_eq!(
        execute(&fixture, "SELECT count() FROM trace_test.spend_logs")
            .await?
            .trim(),
        "1"
    );
    execute(
        &fixture,
        "RENAME TABLE trace_test.unavailable_costs TO trace_test.lens_call_costs",
    )
    .await?;
    deliver(&fixture, vec![cost_row.clone()]).await?;
    deliver(&fixture, vec![cost_row]).await?;
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    let rows = lookup(&fixture, &connection, access(), "gateway-call").await?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].spend, Some(0.125));
    assert_eq!(
        execute(&fixture, "SELECT count() FROM trace_test.spend_logs")
            .await?
            .trim()
            .parse::<usize>()?,
        expected_raw_rows
    );
    assert_eq!(
        execute(&fixture, "SELECT count() FROM trace_test.spend_logs FINAL")
            .await?
            .trim(),
        "1"
    );
    Ok(())
}

#[rstest]
#[case::response_alias("response_id", "new-response", "provider-response", "new-response")]
#[case::owner("user", "user-b", "gateway-call", "gateway-call")]
#[tokio::test]
async fn canonical_updates_fence_stale_aliases_and_owners(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    cost_row: BTreeMap<String, Value>,
    #[case] field: &str,
    #[case] replacement: &str,
    #[case] old_alias: &str,
    #[case] current_alias: &str,
    #[values(false, true)] scoped: bool,
) -> TestResult {
    let fixture = migrated_database?;
    let newer = cost_row
        .clone()
        .into_iter()
        .chain([
            (field.into(), json!(replacement)),
            ("end_time".into(), json!(START_MS + 2)),
            ("spend".into(), json!(0.5)),
        ])
        .collect();
    deliver(&fixture, vec![cost_row.clone()]).await?;
    deliver(&fixture, vec![newer]).await?;
    deliver(&fixture, vec![cost_row]).await?;
    let old_access = contracts::ReadAccessParams {
        all_teams: false,
        user_id: "user-a".into(),
        team_ids: Vec::new(),
    };
    let connection = if scoped {
        fixture
            .readers
            .connection(
                &fixture.database.client,
                &QueryScope::Owned {
                    user_id: "user-a".into(),
                    team_ids: Vec::new(),
                },
                "test-secret",
            )
            .await?
    } else {
        Connection::reader(&fixture.database.url, DATABASE)?
    };
    assert!(
        lookup(&fixture, &connection, old_access, old_alias)
            .await?
            .is_empty()
    );
    let admin = Connection::reader(&fixture.database.url, DATABASE)?;
    let current = lookup(&fixture, &admin, access(), current_alias).await?;
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].spend, Some(0.5));
    Ok(())
}

#[rstest]
#[case::gateway("gateway-call")]
#[case::legacy_gateway("billing-record")]
#[case::provider_request("provider-request")]
#[case::managed_response("upstream-response")]
#[case::transport("transport-trace")]
#[tokio::test]
async fn historical_raw_rows_require_explicit_projection_with_legacy_iso_timestamps(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    mut cost_row: BTreeMap<String, Value>,
    #[case] alias: &str,
) -> TestResult {
    let fixture = migrated_database?;
    if alias == "billing-record" {
        cost_row.remove("litellm_call_id");
    }
    let timestamp =
        time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(START_MS) * 1_000_000)?
            .format(&time::format_description::well_known::Rfc3339)?;
    let row: BTreeMap<String, Value> = cost_row
        .into_iter()
        .chain([
            ("start_time".into(), json!(timestamp)),
            ("end_time".into(), json!(timestamp)),
            (
                "response_id".into(),
                json!(format!(
                    "resp_{}",
                    STANDARD.encode("model_id:deployment;response_id:upstream-response")
                )),
            ),
        ])
        .collect();
    execute(&fixture, &format!("INSERT INTO trace_test.spend_logs SETTINGS date_time_input_format='best_effort' FORMAT JSONEachRow\n{}", encode_rows(vec![row.clone()])?)).await?;
    let connection = Connection::reader(&fixture.database.url, DATABASE)?;
    assert!(
        lookup(&fixture, &connection, access(), alias)
            .await?
            .is_empty()
    );
    let writer = Connection::writer(&fixture.database.url)?;
    project_spend_rows(&fixture.database.client, &writer, DATABASE, vec![row]).await?;
    let projected = lookup(&fixture, &connection, access(), alias).await?;
    assert_eq!(projected.len(), 1);
    assert_eq!(
        (projected[0].spend, projected[0].start_ms),
        (Some(0.125), START_MS)
    );
    assert_eq!(
        execute(&fixture, "SELECT count() FROM trace_test.spend_logs")
            .await?
            .trim(),
        "1"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn invalid_projection_is_rejected_before_raw_persistence(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
    cost_row: BTreeMap<String, Value>,
) -> TestResult {
    let fixture = migrated_database?;
    let row = cost_row
        .into_iter()
        .chain([("spend".into(), json!("0.125"))])
        .collect();
    assert!(deliver(&fixture, vec![row]).await.is_err());
    assert_eq!(
        execute(&fixture, "SELECT count() FROM trace_test.spend_logs")
            .await?
            .trim(),
        "0"
    );
    Ok(())
}
