//! The flusher: statement splitting, poison-row isolation and retries for one pass of spend logs,
//! as the Python `update_spend_logs` does them. Database failures are classified by
//! `litellm_db::Error::kind`, as every other Rust database caller does.

use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use litellm_db::{Error, ErrorKind};
use litellm_error::Report;
use litellm_repositories::{LedgerCommit, SpendLogWrite};

use crate::{
    QueuedSpendLog, Retention, WriterSettings,
    committed::{CommitObserver, committed_rows},
    statement::{delivery, spend_log_write},
};

const STATEMENT_FRAMING_BYTES: usize = 2;
const ROW_SEPARATOR_BYTES: usize = 2;

/// Where a statement's writes are committed: Postgres in production, a fake in tests.
pub trait LedgerStore: Send + Sync {
    fn insert(&self, writes: &[SpendLogWrite])
    -> impl Future<Output = Result<usize, Error>> + Send;

    /// Like `insert`, also naming the deliveries its committed transaction accepted. A store
    /// that cannot prove which deliveries it accepted confirms none.
    fn insert_committed(
        &self,
        writes: &[SpendLogWrite],
    ) -> impl Future<Output = Result<LedgerCommit, Error>> + Send {
        async move {
            Ok(LedgerCommit {
                count: self.insert(writes).await?,
                accepted: Vec::new(),
            })
        }
    }
}

/// Worth retrying in the same flush: the database dropped or refused the connection, is
/// restarting, or picked this statement as a deadlock victim.
fn is_transient(error: &Error) -> bool {
    matches!(error.kind(), ErrorKind::Unreachable | ErrorKind::Deadlock)
}

/// Caused by the rows themselves, so isolation narrows it to the bad row.
fn is_rejected_row(error: &Error) -> bool {
    error.kind() == ErrorKind::DataRejected
}

/// Splits rows of the given sizes into statements of at most `max_rows` rows and `max_bytes`
/// bytes, counting the array framing; a row over the budget goes alone.
fn write_batches(sizes: &[usize], max_bytes: usize, max_rows: usize) -> Vec<(usize, usize)> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < sizes.len() {
        let mut end = start + 1;
        let mut used = STATEMENT_FRAMING_BYTES + sizes[start];
        while end < sizes.len()
            && end - start < max_rows
            && used + ROW_SEPARATOR_BYTES + sizes[end] <= max_bytes
        {
            used += ROW_SEPARATOR_BYTES + sizes[end];
            end += 1;
        }
        batches.push((start, end));
        start = end;
    }
    batches
}

fn charges_of(writes: &[SpendLogWrite]) -> Vec<SpendLogWrite> {
    writes
        .iter()
        .filter_map(|write| {
            write.delivery.clone().map(|delivery| SpendLogWrite {
                log: None,
                delivery: Some(delivery),
            })
        })
        .collect()
}

fn request_id(write: &SpendLogWrite) -> &str {
    match (&write.log, &write.delivery) {
        (Some(log), _) => &log.request_id,
        (None, Some(delivery)) => &delivery.request_id,
        (None, None) => "None",
    }
}

/// Writes a pass of rows: statements within the byte and row budgets, a row the table cannot
/// hold dropped alone, bisection around rows Postgres rejects, and up to `write_retries` retries
/// with `1 << attempt` second sleeps on transient failures. A dropped row keeps its charge.
pub struct SpendLogFlusher<S> {
    store: S,
    settings: WriterSettings,
    observer: Option<CommitObserver>,
    reporting: AtomicBool,
}

type Isolation<'a> = Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + 'a>>;

impl<S: LedgerStore> SpendLogFlusher<S> {
    pub fn new(store: S, settings: WriterSettings) -> Self {
        Self {
            store,
            settings,
            observer: None,
            reporting: AtomicBool::new(false),
        }
    }

    /// Turns commit reports on or off. Reports start off, so a writer whose observer has no
    /// listener builds no summaries.
    pub fn set_reporting(&self, enabled: bool) {
        self.reporting.store(enabled, Ordering::Release);
    }

    pub fn reporting(&self) -> bool {
        self.reporting.load(Ordering::Acquire)
    }

    /// Tells `observer` about each committed statement's accepted deliveries, after the commit.
    pub fn with_commit_observer(mut self, observer: CommitObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Commits `writes` as one statement and reports what it accepted. Returns the store count.
    pub async fn commit(&self, writes: &[SpendLogWrite]) -> Result<usize, Error> {
        let committed = self.store.insert_committed(writes).await?;
        if let Some(observer) = &self.observer
            && !committed.accepted.is_empty()
            && self.reporting()
        {
            let report = std::panic::AssertUnwindSafe(|| {
                observer(committed_rows(writes, &committed.accepted));
            });
            if std::panic::catch_unwind(report).is_err() {
                tracing::warn!(
                    "Spend tracking - the commit observer panicked; accounting is unaffected"
                );
            }
        }
        Ok(committed.count)
    }

    pub fn settings(&self) -> &WriterSettings {
        &self.settings
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    /// Writes `rows`, retrying transient failures. On an error nothing tells which statements
    /// committed, so the caller requeues every row; a replay is safe because each delivery and
    /// each log is inserted at most once.
    pub async fn write(
        &self,
        rows: &[QueuedSpendLog],
        retention: Option<&Retention>,
    ) -> Result<(), Error> {
        if !rows.is_empty() {
            tracing::info!(
                "Spend tracking - processing {} spend logs for DB write",
                rows.len()
            );
        }
        let retries = self.settings.write_retries;
        for attempt in 0..=retries {
            let Err(error) = self.write_once(rows, retention).await else {
                return Ok(());
            };
            if !is_transient(&error) {
                tracing::warn!(
                    "Spend tracking - DB error writing spend logs, requeued {} rows for the next flush. error={}",
                    rows.len(),
                    Report::new(&error)
                );
                return Err(error);
            }
            tracing::warn!(
                "Spend tracking - transient DB error writing spend logs, retry {}/{}. logs_count={}, error={}",
                attempt + 1,
                retries,
                rows.len(),
                Report::new(&error)
            );
            if attempt >= retries {
                return Err(error);
            }
            tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
        }
        Ok(())
    }

    async fn write_once(
        &self,
        rows: &[QueuedSpendLog],
        retention: Option<&Retention>,
    ) -> Result<(), Error> {
        for slice in rows.chunks(self.settings.isolation_slice_rows.max(1)) {
            self.write_slice(slice, retention).await?;
        }
        tracing::debug!("Flushed {} logs to the DB.", rows.len());
        Ok(())
    }

    /// Writes one slice of the pass with a fresh isolation failure budget, so rows Postgres
    /// rejects in one slice never cost the isolation of another.
    async fn write_slice(
        &self,
        rows: &[QueuedSpendLog],
        retention: Option<&Retention>,
    ) -> Result<(), Error> {
        let sizes: Vec<usize> = rows.iter().map(QueuedSpendLog::bytes).collect();
        let mut budget = self.settings.isolation_failure_budget;
        for (start, end) in write_batches(
            &sizes,
            self.settings.write_batch_max_bytes,
            self.settings.write_batch_max_rows,
        ) {
            let now = chrono::Utc::now().naive_utc();
            let mut writes = Vec::with_capacity(end - start);
            let mut refused = Vec::new();
            for row in &rows[start..end] {
                match spend_log_write(row, retention, now) {
                    Ok(write) => writes.push(write),
                    Err(error) => {
                        tracing::error!(
                            "Spend tracking - dropping spend log row Postgres rejected. request_id={} error={}",
                            row.request_id(),
                            error
                        );
                        refused.extend(delivery(row).map(|delivery| SpendLogWrite {
                            log: None,
                            delivery: Some(delivery),
                        }));
                    }
                }
            }
            self.keep_charges(refused).await?;
            if !writes.is_empty() {
                budget = self.isolate(&writes, budget).await?;
            }
        }
        Ok(())
    }

    async fn keep_charges(&self, charges: Vec<SpendLogWrite>) -> Result<(), Error> {
        if charges.is_empty() {
            return Ok(());
        }
        match self.commit(&charges).await {
            Ok(_) => Ok(()),
            Err(error) if is_rejected_row(&error) => {
                tracing::error!(
                    "Spend tracking - could not keep the charges of {} dropped spend log rows. error={}",
                    charges.len(),
                    Report::new(&error)
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn isolate<'a>(&'a self, writes: &'a [SpendLogWrite], failure_budget: usize) -> Isolation<'a> {
        Box::pin(async move {
            let error = match self.commit(writes).await {
                Ok(_) => return Ok(failure_budget),
                Err(error) => error,
            };
            if !is_rejected_row(&error) {
                return Err(error);
            }
            let budget_left = failure_budget.saturating_sub(1);
            if writes.len() == 1 {
                tracing::error!(
                    "Spend tracking - dropping spend log row Postgres rejected. request_id={} error={}",
                    request_id(&writes[0]),
                    Report::new(&error)
                );
                self.keep_charges(charges_of(writes)).await?;
                return Ok(budget_left);
            }
            if budget_left == 0 {
                tracing::error!(
                    "Spend tracking - dropping {} spend log rows without per-row isolation; isolation failure budget exhausted for this flush",
                    writes.len()
                );
                self.keep_charges(charges_of(writes)).await?;
                return Ok(0);
            }
            let mid = writes.len() / 2;
            let remaining = self.isolate(&writes[..mid], budget_left).await?;
            if remaining == 0 {
                tracing::error!(
                    "Spend tracking - dropping {} spend log rows without per-row isolation; isolation failure budget exhausted for this flush",
                    writes.len() - mid
                );
                self.keep_charges(charges_of(&writes[mid..])).await?;
                return Ok(0);
            }
            self.isolate(&writes[mid..], remaining).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use litellm_db::Error;
    use rstest::rstest;

    use super::{SpendLogFlusher, write_batches};
    use crate::{
        QueuedSpendLog, WriterSettings,
        test_support::{
            FakeStore, charged, charges_only, connect_error, full_pool, python_row, request_ids,
            rows, sqlstate,
        },
    };

    fn flusher(store: FakeStore) -> SpendLogFlusher<FakeStore> {
        SpendLogFlusher::new(store, WriterSettings::default())
    }

    #[rstest]
    #[case::connect_error(connect_error as fn() -> Error)]
    #[case::server_terminated(|| sqlstate("57P01"))]
    #[case::connection_lost(|| sqlstate("08006"))]
    #[case::deadlock(|| sqlstate("40P01"))]
    #[tokio::test(start_paused = true)]
    async fn retries_transient_errors_until_one_attempt_succeeds(#[case] error: fn() -> Error) {
        let failures = Arc::new(Mutex::new(0usize));
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| {
            let mut failures = failures.lock().expect("failures lock");
            *failures += 1;
            (*failures <= 3).then(error)
        })));
        let started = tokio::time::Instant::now();

        flusher
            .write(&rows(2), None)
            .await
            .expect("fourth attempt writes");

        assert_eq!(flusher.store.calls().statements.len(), 4);
        assert_eq!(started.elapsed(), Duration::from_secs(1 + 2 + 4));
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn gives_up_on_an_outage_after_three_retries() {
        let flusher = flusher(FakeStore::new(Box::new(|_, _| Some(connect_error()))));
        let started = tokio::time::Instant::now();

        let error = flusher
            .write(&rows(2), None)
            .await
            .expect_err("never writes");

        assert!(error.is_unavailable());
        assert_eq!(flusher.store.calls().statements.len(), 4);
        assert_eq!(started.elapsed(), Duration::from_secs(1 + 2 + 4));
    }

    #[rstest]
    #[case::systemic_database_error(|| sqlstate("42P01"))]
    #[case::capacity(|| sqlstate("53300"))]
    #[case::full_pool(full_pool as fn() -> Error)]
    #[case::transaction_timeout(|| Error::TransactionTimedOut)]
    #[tokio::test(start_paused = true)]
    async fn a_non_transient_error_fails_on_the_first_attempt(#[case] error: fn() -> Error) {
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| Some(error()))));

        let failure = flusher.write(&rows(2), None).await.expect_err("fails once");

        assert!(failure.is_requeue_safe());
        assert_eq!(flusher.store.calls().statements.len(), 1);
    }

    #[rstest]
    #[case::one_and_a_half_statements(150)]
    #[case::many_statements(2000)]
    #[tokio::test]
    async fn splits_the_pass_into_bounded_statements_in_order(#[case] count: usize) {
        let flusher = flusher(FakeStore::writing());

        flusher.write(&rows(count), None).await.expect("writes");

        let calls = flusher.store.calls();
        assert_eq!(calls.statements.len(), count.div_ceil(100));
        assert!(
            calls
                .statements
                .iter()
                .all(|statement| statement.len() <= 100)
        );
        assert_eq!(
            calls.written,
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn a_transient_failure_replays_the_whole_pass() {
        let flusher = flusher(FakeStore::new(Box::new(|_, call| {
            (call == 2).then(connect_error)
        })));

        flusher.write(&rows(400), None).await.expect("writes");

        let calls = flusher.store.calls();
        assert_eq!(calls.statements.len(), 2 + 4);
        let mut written = calls.written.clone();
        written.sort();
        written.dedup();
        assert_eq!(written.len(), 400);
    }

    #[rstest]
    #[case::data_exception("22P02")]
    #[case::not_null("23502")]
    #[case::too_large("54000")]
    #[tokio::test]
    async fn a_row_postgres_rejects_is_dropped_alone_and_keeps_its_charge(
        #[case] code: &'static str,
    ) {
        let flusher = flusher(FakeStore::new(Box::new(move |writes, _| {
            (!charges_only(writes) && request_ids(writes).iter().any(|id| id == "rejected"))
                .then(|| sqlstate(code))
        })));
        let batch = [
            rows(1).remove(0),
            charged("rejected"),
            QueuedSpendLog::from_python(&python_row("good", "")).expect("queued"),
        ];

        flusher.write(&batch, None).await.expect("writes the rest");

        let calls = flusher.store.calls();
        assert_eq!(calls.written, ["0", "good"]);
        assert_eq!(calls.charges_kept, ["rejected"]);
    }

    #[tokio::test]
    async fn the_commit_observer_hears_the_accepted_accounting_of_kept_and_isolated_rows() {
        let reported: Arc<Mutex<Vec<serde_json::Value>>> = Arc::default();
        let sink = Arc::clone(&reported);
        let flusher = SpendLogFlusher::new(
            FakeStore::new(Box::new(move |writes, _| {
                (!charges_only(writes) && request_ids(writes).iter().any(|id| id == "rejected"))
                    .then(|| sqlstate("22P02"))
            })),
            WriterSettings::default(),
        )
        .with_commit_observer(Arc::new(move |rows: Vec<String>| {
            sink.lock().expect("reported lock").extend(
                rows.iter()
                    .map(|text| serde_json::from_str::<serde_json::Value>(text).expect("json")),
            );
        }));
        flusher.set_reporting(true);

        flusher
            .write(&[charged("accepted"), charged("rejected")], None)
            .await
            .expect("writes the rest");

        let mut reported = reported.lock().expect("reported lock").clone();
        reported.sort_by_key(|row| row["delivery_id"].to_string());
        assert_eq!(
            reported
                .iter()
                .map(|row| (row["delivery_id"].clone(), row["spend"].clone()))
                .collect::<Vec<_>>(),
            [
                (serde_json::json!("accepted-d"), serde_json::json!(1.0)),
                (serde_json::json!("rejected-d"), serde_json::json!(1.0)),
            ]
        );
        assert!(
            reported
                .iter()
                .all(|row| row.get("endTime").is_none() && row.get("model_id").is_none())
        );
        assert!(reported.iter().all(|row| row.get("messages").is_none()));
    }

    #[tokio::test]
    async fn a_store_that_cannot_prove_acceptance_reports_nothing() {
        struct CountOnly;
        impl crate::LedgerStore for CountOnly {
            async fn insert(
                &self,
                writes: &[litellm_repositories::SpendLogWrite],
            ) -> Result<usize, Error> {
                Ok(writes.len())
            }
        }
        let reported: Arc<Mutex<usize>> = Arc::default();
        let sink = Arc::clone(&reported);
        let flusher = SpendLogFlusher::new(CountOnly, WriterSettings::default())
            .with_commit_observer(Arc::new(move |_| *sink.lock().expect("lock") += 1));
        flusher.set_reporting(true);

        flusher.write(&[charged("a")], None).await.expect("writes");

        assert_eq!(*reported.lock().expect("lock"), 0);
    }

    fn write_of(row: &QueuedSpendLog) -> litellm_repositories::SpendLogWrite {
        crate::statement::spend_log_write(row, None, chrono::Utc::now().naive_utc())
            .expect("row writes")
    }

    #[tokio::test]
    async fn reports_start_off_and_follow_the_reporting_switch() {
        let reported: Arc<Mutex<usize>> = Arc::default();
        let sink = Arc::clone(&reported);
        let flusher = flusher(FakeStore::writing()).with_commit_observer(Arc::new(move |rows| {
            *sink.lock().expect("lock") += rows.len();
        }));

        assert!(!flusher.reporting());
        for (name, enabled) in [("off", false), ("on", true), ("off-again", false)] {
            flusher.set_reporting(enabled);
            let written = flusher.commit(&[write_of(&charged(name))]).await;
            assert_eq!(written.expect("commits"), 1);
        }

        assert_eq!(*reported.lock().expect("lock"), 1);
        assert_eq!(flusher.store.calls().written, ["off", "on", "off-again"]);
    }

    #[rstest]
    #[case::unparseable_json_column(r#", "messages": "not json""#)]
    #[case::malformed_timestamp(r#", "startTime": "soon""#)]
    #[tokio::test]
    async fn a_row_the_table_cannot_hold_is_dropped_alone_before_postgres(#[case] fields: &str) {
        let flusher = flusher(FakeStore::writing());
        let malformed = QueuedSpendLog::from_python(&python_row(
            "malformed",
            &format!(
                r#"{fields}, "accounting_delivery_id": "d", "accounting_payload": {{"spend": 1.0}}"#
            ),
        ))
        .expect("queued");

        flusher
            .write(&[charged("good"), malformed], None)
            .await
            .expect("writes the rest");

        let calls = flusher.store.calls();
        assert_eq!(calls.written, ["good"]);
        assert_eq!(calls.charges_kept, ["malformed"]);
    }

    #[rstest]
    #[case::undefined_table("42P01")]
    #[case::deadlock("40P01")]
    #[tokio::test]
    async fn a_systemic_failure_is_not_isolated(#[case] code: &'static str) {
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| Some(sqlstate(code)))));
        let writes = vec![
            crate::statement::spend_log_write(&charged("a"), None, chrono::Utc::now().naive_utc())
                .expect("write"),
        ];

        assert!(flusher.isolate(&writes, 10).await.is_err());
        assert!(flusher.store.calls().written.is_empty());
    }

    #[rstest]
    #[case::transient_rescue_failure("40P01", Err(()))]
    #[case::rejected_rescue_failure("22P02", Ok(9))]
    #[tokio::test]
    async fn a_failed_charge_rescue_is_logged_unless_it_can_be_retried(
        #[case] rescue_code: &'static str,
        #[case] expected: Result<usize, ()>,
    ) {
        let flusher = flusher(FakeStore::new(Box::new(move |writes, _| {
            Some(sqlstate(if charges_only(writes) {
                rescue_code
            } else {
                "22P02"
            }))
        })));
        let writes = vec![
            crate::statement::spend_log_write(
                &charged("rejected"),
                None,
                chrono::Utc::now().naive_utc(),
            )
            .expect("write"),
        ];

        assert_eq!(flusher.isolate(&writes, 10).await.map_err(|_| ()), expected);
    }

    #[rstest]
    #[tokio::test]
    async fn every_slice_of_the_pass_isolates_with_a_fresh_failure_budget() {
        let flusher = SpendLogFlusher::new(
            FakeStore::new(Box::new(|writes, _| {
                (!charges_only(writes) && request_ids(writes).iter().any(|id| id.starts_with('x')))
                    .then(|| sqlstate("22P02"))
            })),
            WriterSettings {
                write_batch_max_rows: 2,
                isolation_failure_budget: 3,
                isolation_slice_rows: 2,
                ..WriterSettings::default()
            },
        );
        let pass: Vec<_> = ["x0", "a", "x1", "b"].into_iter().map(charged).collect();

        flusher.write(&pass, None).await.expect("writes");

        let calls = flusher.store.calls();
        assert_eq!(calls.written, ["a", "b"]);
        assert_eq!(calls.charges_kept, ["x0", "x1"]);
    }

    #[rstest]
    #[case::budget_spent_on_the_first_half(1)]
    #[tokio::test]
    async fn an_exhausted_budget_drops_the_rest_and_keeps_their_charges(#[case] budget: usize) {
        let flusher = flusher(FakeStore::new(Box::new(|writes, _| {
            (!charges_only(writes)).then(|| sqlstate("22P02"))
        })));
        let now = chrono::Utc::now().naive_utc();
        let writes: Vec<_> = [charged("a"), charged("b")]
            .iter()
            .map(|row| crate::statement::spend_log_write(row, None, now).expect("write"))
            .collect();

        assert_eq!(flusher.isolate(&writes, budget).await.expect("drops"), 0);
        assert_eq!(flusher.store.calls().charges_kept, ["a", "b"]);
    }

    #[rstest]
    #[case::row_budget(&[10, 10, 10], 1000, 2, vec![(0, 2), (2, 3)])]
    #[case::byte_budget_counts_framing(&[10, 10, 10], 2 + 10 + 2 + 10, 100, vec![(0, 2), (2, 3)])]
    #[case::oversized_row_alone(&[5000, 10], 100, 100, vec![(0, 1), (1, 2)])]
    fn statements_respect_both_budgets(
        #[case] sizes: &[usize],
        #[case] max_bytes: usize,
        #[case] max_rows: usize,
        #[case] expected: Vec<(usize, usize)>,
    ) {
        assert_eq!(write_batches(sizes, max_bytes, max_rows), expected);
    }
}
