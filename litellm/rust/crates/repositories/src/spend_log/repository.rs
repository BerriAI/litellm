//! The spend log repository: the spend a key, team or user logged since a window start, read when a budget window
//! counter is cold and the maintained `LiteLLM_BudgetWindowSpend` row cannot answer
//!
//! It also writes the logs: one ledger commit records each request's accounting delivery in
//! `LiteLLM_SpendLogDelivery` and its request log in `LiteLLM_SpendLogs`, so a replayed flush
//! can neither charge a request twice nor lose its log.
//!
//! The policy mirrors `SpendLogsRepository.insert_with_deliveries`: logs with no charge go in
//! on their own; deliveries go in first with `ON CONFLICT DO NOTHING`, a batch or video job
//! that a pre-ledger cost row already charged loses its delivery again, and only the logs whose
//! delivery this commit accepted are written, all inside one transaction with its own deadline.

use std::{collections::BTreeMap, time::Duration};

use chrono::{DateTime, Utc};
use litellm_db::Call;
use litellm_db::{Error, Statements};
use serde::Serialize;
use serde_json::{Value, value::RawValue};
use sqlx::{PgConnection, types::Json};

use super::queries;
use super::record::{SpendLogDelivery, SpendLogRecord, SpendLogWrite};
use crate::{Database, Side};

const WINDOW_SPEND: Call = Call {
    call_type: "group_by",
    table: Some("LiteLLM_SpendLogs"),
    operation: Some("group_by"),
};

const INSERT_SPEND_LOGS: Call = Call {
    call_type: "insert_spend_logs",
    table: Some("LiteLLM_SpendLogs"),
    operation: None,
};

/// Whose spend a window sum adds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpendLogOwner {
    /// `api_key`, the hashed virtual key
    Key,
    Team,
    User,
}

/// Reads `LiteLLM_SpendLogs`.
///
/// Also writes request logs and their ledger deliveries on the writer.
pub struct SpendLogRepository {
    database: Database,
}

impl SpendLogRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// The total spend `owner_id` logged at or after `since`, zero when it logged none.
    pub async fn window_spend(
        &self,
        owner: SpendLogOwner,
        owner_id: &str,
        since: DateTime<Utc>,
    ) -> Result<f64, litellm_db::Error> {
        let since = since.naive_utc();
        self.database
            .run(WINDOW_SPEND, || async {
                let mut session = self.database.session(Side::Read).await?;
                queries::window_spend(session.executor(), owner, owner_id, since).await
            })
            .await
    }

    /// Writes one statement's worth of requests as one reported operation: logs with no delivery
    /// go in on their own, the rest commit in one ledger transaction that gives up after
    /// `transaction_timeout`. Returns how many logs and deliveries were new.
    pub async fn insert_with_deliveries(
        &self,
        writes: &[SpendLogWrite],
        transaction_timeout: Duration,
    ) -> Result<usize, Error> {
        self.database
            .run(INSERT_SPEND_LOGS, || {
                self.write(writes, transaction_timeout)
            })
            .await
    }

    async fn write(
        &self,
        writes: &[SpendLogWrite],
        transaction_timeout: Duration,
    ) -> Result<usize, Error> {
        let logs: Vec<&SpendLogRecord> = writes
            .iter()
            .filter(|write| write.delivery.is_none())
            .filter_map(|write| write.log.as_ref())
            .collect();
        let ledger: Vec<&SpendLogWrite> = writes
            .iter()
            .filter(|write| write.delivery.is_some())
            .collect();
        let mut session = self.database.session(Side::Writer).await?;
        let logged = if logs.is_empty() {
            0
        } else {
            queries::insert_spend_log_rows(session.executor(), json_text(&logs)?)
                .await?
                .len()
        };
        if ledger.is_empty() {
            return Ok(logged);
        }
        let named = session.names_statements();
        let accepted =
            commit_ledger(session.connection(), named, &ledger, transaction_timeout).await?;
        Ok(logged + accepted)
    }
}

/// Commits `writes` on `connection` as one transaction, giving up with `TransactionTimedOut`
/// after `timeout` so a stuck commit cannot hold the flush forever.
async fn commit_ledger(
    connection: &mut PgConnection,
    named: bool,
    writes: &[&SpendLogWrite],
    timeout: Duration,
) -> Result<usize, Error> {
    let commit = async {
        let mut transaction = sqlx::Connection::begin(connection).await?;
        let accepted = write_ledger(&mut transaction, named, writes).await?;
        transaction.commit().await?;
        Ok(accepted)
    };
    tokio::time::timeout(timeout, commit)
        .await
        .unwrap_or(Err(Error::TransactionTimedOut))
}

/// `rows` as the JSON array `jsonb_populate_recordset` and `jsonb_to_recordset` read.
fn json(rows: &impl Serialize) -> Result<Value, Error> {
    serde_json::to_value(rows).map_err(|error| Error::Query(sqlx::Error::Encode(Box::new(error))))
}

/// `records` as the JSON array text `jsonb_populate_recordset` reads, keeping each JSON column's
/// text as written.
fn json_text(records: &[&SpendLogRecord]) -> Result<Json<Box<RawValue>>, Error> {
    serde_json::to_string(records)
        .and_then(RawValue::from_string)
        .map(Json)
        .map_err(|error| Error::Query(sqlx::Error::Encode(Box::new(error))))
}

/// The first write per delivery id, ordered by delivery id so concurrent flushes lock delivery
/// rows in the same order.
fn first_write_per_delivery<'a>(
    writes: &[&'a SpendLogWrite],
) -> Vec<(&'a SpendLogDelivery, &'a SpendLogWrite)> {
    let mut by_delivery: BTreeMap<&str, (&SpendLogDelivery, &SpendLogWrite)> = BTreeMap::new();
    for write in writes {
        if let Some(delivery) = &write.delivery {
            by_delivery
                .entry(delivery.delivery_id.as_str())
                .or_insert((delivery, write));
        }
    }
    by_delivery.into_values().collect()
}

/// Writes `writes` on `connection`, which holds an open transaction: the first write per
/// delivery id wins, a delivery already stored is skipped, and only the logs of accepted
/// deliveries are written. Returns how many deliveries were accepted.
async fn write_ledger(
    connection: &mut PgConnection,
    named: bool,
    writes: &[&SpendLogWrite],
) -> Result<usize, Error> {
    let ledger = first_write_per_delivery(writes);
    if ledger.is_empty() {
        return Ok(0);
    }
    let deliveries: Vec<&SpendLogDelivery> = ledger.iter().map(|(delivery, _)| *delivery).collect();
    let accepted = queries::insert_spend_log_deliveries(
        Statements::new(&mut *connection, named),
        json(&deliveries)?,
    )
    .await?;
    let settled_once: Vec<String> = deliveries
        .iter()
        .filter(|delivery| delivery.is_batch && accepted.contains(&delivery.delivery_id))
        .map(|delivery| delivery.delivery_id.clone())
        .collect();
    let charged_by_legacy = if settled_once.is_empty() {
        Vec::new()
    } else {
        queries::spend_log_jobs_charged_by_legacy(
            Statements::new(&mut *connection, named),
            &settled_once,
        )
        .await?
    };
    if !charged_by_legacy.is_empty() {
        queries::delete_spend_log_deliveries(
            Statements::new(&mut *connection, named),
            &charged_by_legacy,
        )
        .await?;
    }
    let kept = |delivery_id: &String| {
        accepted.contains(delivery_id) && !charged_by_legacy.contains(delivery_id)
    };
    let logs: Vec<&SpendLogRecord> = ledger
        .iter()
        .filter(|(delivery, _)| kept(&delivery.delivery_id))
        .filter_map(|(_, write)| write.log.as_ref())
        .collect();
    if !logs.is_empty() {
        queries::insert_spend_log_rows(Statements::new(&mut *connection, named), json_text(&logs)?)
            .await?;
    }
    Ok(accepted.iter().filter(|id| kept(id)).count())
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, time::Duration};

    use chrono::NaiveDateTime;
    use litellm_db::{Error, ErrorKind, Statements};
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};
    use sqlx::{PgExecutor, PgPool};

    use super::{commit_ledger, first_write_per_delivery, json_text, queries, write_ledger};
    use crate::{
        spend_log::record::{JsonText, SpendLogDelivery, SpendLogRecord, SpendLogWrite},
        test_support::database_pool,
    };

    fn at(text: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").expect("timestamp parses")
    }

    fn text(value: Value) -> JsonText {
        JsonText::new(value.to_string()).expect("JSON text")
    }

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    fn row(request_id: &str) -> SpendLogRecord {
        SpendLogRecord {
            request_id: request_id.into(),
            call_type: "acompletion".into(),
            api_key: "hashed".into(),
            spend: 0.5,
            total_tokens: 3,
            prompt_tokens: 1,
            completion_tokens: 2,
            start_time: at("2026-11-02 12:00:00"),
            end_time: at("2026-11-02 12:00:01"),
            request_duration_ms: Some(1000),
            completion_start_time: None,
            model: "gpt".into(),
            model_id: Some(String::new()),
            model_group: Some(String::new()),
            custom_llm_provider: Some("openai".into()),
            api_base: Some(String::new()),
            user: Some(String::new()),
            metadata: Some(text(json!({"accounting_savings": {"compression": 0.2}}))),
            cache_hit: Some("False".into()),
            cache_key: Some(String::new()),
            request_tags: Some(text(json!(["a"]))),
            team_id: None,
            organization_id: None,
            end_user: None,
            requester_ip_address: None,
            messages: Some(text(json!([{"role": "user", "content": "hi"}]))),
            response: Some(text(json!({}))),
            session_id: None,
            status: Some("success".into()),
            mcp_namespaced_tool_name: None,
            billing_agent_id: None,
            agent_id: None,
            proxy_server_request: Some(text(json!({}))),
            litellm_call_id: None,
            created_at: at("2026-11-02 12:00:02"),
            updated_at: at("2026-11-02 12:00:02"),
        }
    }

    fn write(delivery_id: &str, request_id: &str, logged: bool) -> SpendLogWrite {
        SpendLogWrite {
            log: logged.then(|| row(request_id)),
            delivery: Some(SpendLogDelivery {
                delivery_id: delivery_id.into(),
                request_id: request_id.into(),
                accounting_payload: object(json!({"request_id": request_id, "spend": 0.5})),
                is_batch: false,
            }),
        }
    }

    #[fixture]
    fn ledger() -> Vec<SpendLogWrite> {
        vec![
            write("delivery-b", "ledger-b", true),
            write("delivery-a", "ledger-a", true),
            write("delivery-a", "ledger-a-replayed", true),
            write("delivery-c", "ledger-c", false),
        ]
    }

    #[rstest]
    fn the_first_write_per_delivery_wins_in_delivery_order(ledger: Vec<SpendLogWrite>) {
        let writes: Vec<&SpendLogWrite> = ledger.iter().collect();

        let chosen: Vec<(&str, &str)> = first_write_per_delivery(&writes)
            .into_iter()
            .map(|(delivery, _)| (delivery.delivery_id.as_str(), delivery.request_id.as_str()))
            .collect();

        assert_eq!(
            chosen,
            [
                ("delivery-a", "ledger-a"),
                ("delivery-b", "ledger-b"),
                ("delivery-c", "ledger-c")
            ]
        );
    }

    #[rstest]
    fn rows_bind_under_their_column_names() {
        let encoded: Value =
            serde_json::from_str(json_text(&[&row("r")]).expect("encodes").0.get())
                .expect("parses");

        assert_eq!(encoded[0]["startTime"], json!("2026-11-02T12:00:00"));
        assert_eq!(encoded[0]["completionStartTime"], Value::Null);
        assert_eq!(encoded[0]["created_at"], json!("2026-11-02T12:00:02"));
        assert_eq!(encoded[0]["request_tags"], json!(["a"]));
    }

    async fn ids(
        executor: impl PgExecutor<'_>,
        query: &'static str,
        ids: &[&str],
    ) -> Result<Vec<String>, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        sqlx::query_scalar::<_, String>(sqlx::AssertSqlSafe(query))
            .bind(ids)
            .fetch_all(executor)
            .await
    }

    const DELIVERIES: &str = r#"SELECT delivery_id FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = ANY($1) ORDER BY 1"#;
    const LOGS: &str =
        r#"SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1) ORDER BY 1"#;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_json_column_keeps_every_digit_of_an_integer_past_64_bits(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let log = SpendLogRecord {
            metadata: Some(JsonText::new(r#"{"n":18446744073709551616}"#.to_owned())?),
            ..row("big")
        };
        queries::insert_spend_log_rows(Statements::new(&mut *tx, true), json_text(&[&log])?)
            .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let stored = sqlx::query_scalar::<_, String>(sqlx::AssertSqlSafe(
            r#"SELECT metadata::text FROM "LiteLLM_SpendLogs" WHERE request_id = 'big'"#,
        ))
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(stored, r#"{"n": 18446744073709551616}"#);
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn writes_each_delivery_once_with_its_logged_row(
        ledger: Vec<SpendLogWrite>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let writes: Vec<&SpendLogWrite> = ledger.iter().collect();
        assert_eq!(write_ledger(&mut tx, true, &writes).await?, 3);
        assert_eq!(write_ledger(&mut tx, true, &writes).await?, 0);
        let every_id = [
            "delivery-a",
            "delivery-b",
            "delivery-c",
            "ledger-a",
            "ledger-a-replayed",
            "ledger-b",
            "ledger-c",
        ];
        assert_eq!(
            ids(&mut *tx, DELIVERIES, &every_id).await?,
            ["delivery-a", "delivery-b", "delivery-c"]
        );
        assert_eq!(
            ids(&mut *tx, LOGS, &every_id).await?,
            ["ledger-a", "ledger-b"]
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::charged_before_the_ledger(0.25, "success", vec![])]
    #[case::left_a_zero_cost_row(0.0, "success", vec!["job-1"])]
    #[case::left_a_failed_row(0.25, "failure", vec!["job-1"])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_settle_once_job_keeps_its_delivery_unless_a_legacy_row_charged_it(
        #[case] legacy_spend: f64,
        #[case] legacy_status: &str,
        #[case] expected_deliveries: Vec<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let legacy = SpendLogRecord {
            spend: legacy_spend,
            status: Some(legacy_status.to_owned()),
            ..row("job-1")
        };
        queries::insert_spend_log_rows(Statements::new(&mut *tx, true), json_text(&[&legacy])?)
            .await?;
        let job = SpendLogWrite {
            log: Some(row("job-1")),
            delivery: Some(SpendLogDelivery {
                delivery_id: "job-1".into(),
                request_id: "job-1".into(),
                accounting_payload: object(json!({"spend": 0.25})),
                is_batch: true,
            }),
        };
        let accepted = write_ledger(&mut tx, true, &[&job]).await?;
        assert_eq!(accepted, expected_deliveries.len());
        assert_eq!(
            ids(&mut *tx, DELIVERIES, &["job-1"]).await?,
            expected_deliveries
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::non_finite_spend(SpendLogRecord { spend: f64::NAN, ..row("bad") })]
    #[case::nul_in_json(SpendLogRecord { response: Some(text(json!({"text": "a\u{0}b"}))), ..row("bad") })]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_row_postgres_cannot_store_rolls_back_the_whole_ledger(
        #[case] bad: SpendLogRecord,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut outer = pool.begin().await?;
        let mut poisoned = write("delivery-bad", "bad", true);
        poisoned.log = Some(bad);
        let good = write("delivery-good", "good", true);
        let error = commit_ledger(
            &mut outer,
            true,
            &[&good, &poisoned],
            Duration::from_secs(60),
        )
        .await
        .expect_err("the poisoned row must fail the ledger");
        assert_eq!(error.kind(), ErrorKind::DataRejected);
        assert!(
            ids(&mut *outer, DELIVERIES, &["delivery-good", "delivery-bad"])
                .await?
                .is_empty()
        );
        outer.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_ledger_past_its_timeout_commits_nothing(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let held = write("delivery-timeout", "timeout", true);
        let mut holder = pool.begin().await?;
        write_ledger(&mut holder, true, &[&held]).await?;
        let mut waiter = pool.acquire().await?;
        let result = commit_ledger(&mut waiter, true, &[&held], Duration::from_millis(200)).await;
        holder.rollback().await?;
        assert!(matches!(result, Err(Error::TransactionTimedOut)));
        assert!(
            ids(&pool, DELIVERIES, &["delivery-timeout"])
                .await?
                .is_empty()
        );
        waiter.close().await?;
        Ok(())
    }
}
