//! The spend log writer: a byte-bounded queue of request logs and the flusher that commits them
//! to `LiteLLM_SpendLogs` and `LiteLLM_SpendLogDelivery` through the spend log repository's ledger
//! transaction (`litellm-repositories`).
//!
//! It ports the ledger-on half of the Python proxy's spend log path (`enqueue_spend_logs`,
//! `ProxyUpdateSpend.update_spend_logs`, `SpendLogsRepository.insert_with_deliveries`): the same
//! byte budget, statement sizes, poison-row isolation, retries and requeue rules. Rows are typed
//! from the moment they are queued: the Rust callback queues the [`SpendLogRow`] the row builder
//! made, and Python's rows are parsed once when they are queued.
//!
//! The Python proxy's spend log monitor decides when to flush, because the same job parks failed
//! rows in Redis, recovers rows other pods parked, drains the other request-time queues and runs
//! the guardrail usage rollup on the written rows. It calls [`SpendLogWriter::flush`] for one pass
//! straight from this queue, so rows do not travel back to Python; [`SpendLogWriter::request_flush`]
//! wakes that monitor through the callback the proxy registered.
//!
//! | Module | What it holds |
//! |---|---|
//! | `entry` | [`QueuedSpendLog`]: a queued row and its ledger [`Charge`], sized as `json.dumps` writes it |
//! | `queue` | [`SpendLogQueue`]: the byte budget and the overflow that keeps charges |
//! | `statement` | the typed record and delivery a queued row writes, and metadata [`Retention`] |
//! | `flusher` | [`SpendLogFlusher`]: statements, isolation and retries |
//! | `store` | [`PgLedgerStore`]: the repository behind [`LedgerStore`] |
//!
//! [`SpendLogRow`]: litellm_callbacks_spend_log::SpendLogRow

mod entry;
mod flusher;
mod queue;
mod statement;
mod store;
#[cfg(test)]
mod test_support;

use std::time::Duration;

use tokio::sync::Mutex;

pub use entry::{Charge, EntryError, QueuedSpendLog, RejectedRow};
pub use flusher::{LedgerStore, SpendLogFlusher};
pub use queue::SpendLogQueue;
pub use statement::{Retention, RowError};
pub use store::PgLedgerStore;

/// Why a settle-once job row could not be claimed; the caller queues it instead.
#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    #[error(transparent)]
    Row(#[from] RowError),
    #[error(transparent)]
    Database(#[from] litellm_db::Error),
}

/// The writer's limits. The defaults are the Python proxy's constants; the proxy passes the ones
/// it reads from env vars.
#[derive(Clone, Debug, PartialEq)]
pub struct WriterSettings {
    /// `SPEND_LOG_QUEUE_MAX_BYTES`.
    pub queue_max_bytes: usize,
    /// `SPEND_LOG_WRITE_BATCH_MAX_BYTES`.
    pub write_batch_max_bytes: usize,
    /// `SPEND_LOG_WRITE_BATCH_MAX_ROWS`.
    pub write_batch_max_rows: usize,
    /// `MAX_SPEND_LOG_ISOLATION_FAILURES_PER_BATCH`, renewed for every slice of
    /// `isolation_slice_rows` rows.
    pub isolation_failure_budget: usize,
    /// The rows of the pass that share one isolation failure budget and that no statement
    /// crosses: the Python writer's `BATCH_SIZE`.
    pub isolation_slice_rows: usize,
    /// The retries after a transient failure, sleeping `1 << attempt` seconds before each.
    pub write_retries: u32,
    /// How long the ledger transaction may take before it is rolled back.
    pub transaction_timeout: Duration,
}

impl Default for WriterSettings {
    fn default() -> Self {
        Self {
            queue_max_bytes: 64_000_000,
            write_batch_max_bytes: 2_000_000,
            write_batch_max_rows: 100,
            isolation_failure_budget: 256,
            isolation_slice_rows: 1000,
            write_retries: 3,
            transaction_timeout: Duration::from_secs(60),
        }
    }
}

/// The queue and the flusher together, with the lock that keeps one pass in flight and the
/// callback that wakes whoever flushes.
pub struct SpendLogWriter<S> {
    queue: SpendLogQueue,
    flusher: SpendLogFlusher<S>,
    write_lock: Mutex<()>,
    wake: Box<dyn Fn() + Send + Sync>,
}

impl<S: LedgerStore> SpendLogWriter<S> {
    /// A writer whose [`SpendLogWriter::request_flush`] calls `wake`.
    pub fn new(flusher: SpendLogFlusher<S>, wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            queue: SpendLogQueue::default(),
            flusher,
            write_lock: Mutex::new(()),
            wake: Box::new(wake),
        }
    }

    pub fn settings(&self) -> &WriterSettings {
        self.flusher.settings()
    }

    pub fn flusher(&self) -> &SpendLogFlusher<S> {
        &self.flusher
    }

    /// Queues `rows` at the tail, or at the head for rows that must flush first, within the
    /// queue's byte budget. Returns how many of the oldest rows it dropped.
    pub fn enqueue(&self, rows: Vec<QueuedSpendLog>, at_head: bool) -> usize {
        self.queue
            .enqueue(rows, at_head, self.settings().queue_max_bytes)
    }

    /// Queues `rows` within `max_bytes` instead of the configured budget.
    pub fn enqueue_within(
        &self,
        rows: Vec<QueuedSpendLog>,
        at_head: bool,
        max_bytes: usize,
    ) -> usize {
        self.queue.enqueue(rows, at_head, max_bytes)
    }

    /// Writes a settle-once job's cost row at once, outside the queue. Every poll of the job
    /// carries the same delivery, so the insert that lands it owns the job's charge; returns
    /// whether this one did.
    pub async fn claim(
        &self,
        entry: &QueuedSpendLog,
        retention: Option<&Retention>,
    ) -> Result<bool, ClaimError> {
        let write = statement::spend_log_write(entry, retention, chrono::Utc::now().naive_utc())?;
        Ok(self.flusher.store().insert(&[write]).await? == 1)
    }

    /// Asks for a pass now rather than at the next poll, for a row a following request reads back
    /// from the database (a Responses API turn).
    pub fn request_flush(&self) {
        (self.wake)();
    }

    /// Writes one pass of up to `limit` rows from the head of the queue and returns them. A pass
    /// that fails puts its rows back at the head, so a replay writes them first. Passes run one at
    /// a time.
    pub async fn flush(
        &self,
        limit: usize,
        retention: Option<&Retention>,
    ) -> Result<Vec<QueuedSpendLog>, litellm_db::Error> {
        let _pass = self.write_lock.lock().await;
        let rows = self.queue.dequeue(limit);
        if rows.is_empty() {
            return Ok(rows);
        }
        match self.flusher.write(&rows, retention).await {
            Ok(()) => Ok(rows),
            Err(error) => {
                self.enqueue(rows, true);
                Err(error)
            }
        }
    }

    /// Waits for the pass in flight, if any, to finish. A pass keeps running when the task that
    /// awaited it is cancelled, so shutdown waits here before it counts what is still queued.
    pub async fn idle(&self) {
        drop(self.write_lock.lock().await);
    }

    /// Takes up to `limit` rows off the head without writing them, for a caller that parks them
    /// elsewhere.
    pub fn take(&self, limit: usize) -> Vec<QueuedSpendLog> {
        self.queue.dequeue(limit)
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn queued_bytes(&self) -> usize {
        self.queue.bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use rstest::rstest;

    use super::{SpendLogFlusher, SpendLogWriter, WriterSettings};
    use crate::test_support::{FakeStore, connect_error, full_pool, rows};

    fn writer(store: FakeStore) -> SpendLogWriter<FakeStore> {
        SpendLogWriter::new(
            SpendLogFlusher::new(store, WriterSettings::default()),
            || {},
        )
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn idle_waits_for_a_pass_retrying_in_the_background_to_put_its_rows_back() {
        let writer = Arc::new(writer(FakeStore::new(Box::new(|_, _| {
            Some(connect_error())
        }))));
        writer.enqueue(rows(3), false);
        let pass = tokio::spawn({
            let writer = Arc::clone(&writer);
            async move { writer.flush(3, None).await.map(|_| ()) }
        });
        tokio::task::yield_now().await;
        assert_eq!(writer.len(), 0);

        writer.idle().await;

        assert_eq!(writer.len(), 3);
        assert!(pass.await.expect("pass task").is_err());
    }

    #[rstest]
    #[case::full_pool(full_pool as fn() -> litellm_db::Error)]
    #[case::outage(connect_error as fn() -> litellm_db::Error)]
    #[tokio::test(start_paused = true)]
    async fn a_failed_pass_requeues_its_rows_at_the_head(#[case] error: fn() -> litellm_db::Error) {
        let writer = writer(FakeStore::new(Box::new(move |_, _| Some(error()))));
        writer.enqueue(rows(3), false);

        writer.flush(2, None).await.expect_err("pass fails");

        let order: Vec<String> = writer
            .take(10)
            .iter()
            .map(|row| row.request_id().to_owned())
            .collect();
        assert_eq!(order, ["0", "1", "2"]);
    }

    #[rstest]
    #[case::partial_pass(25, 10, 15)]
    #[case::everything(5, 10, 0)]
    #[tokio::test]
    async fn a_pass_writes_up_to_its_limit_and_returns_the_rows(
        #[case] queued: usize,
        #[case] limit: usize,
        #[case] left: usize,
    ) {
        let writer = writer(FakeStore::writing());
        writer.enqueue(rows(queued), false);

        let written = writer.flush(limit, None).await.expect("writes");

        assert_eq!(written.len(), queued - left);
        assert_eq!(writer.len(), left);
        assert_eq!(
            writer.flusher().store().calls().written.len(),
            queued - left
        );
    }

    #[rstest]
    fn a_flush_request_calls_the_waker() {
        let woken = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&woken);
        let writer = SpendLogWriter::new(
            SpendLogFlusher::new(FakeStore::writing(), WriterSettings::default()),
            move || {
                counted.fetch_add(1, Ordering::SeqCst);
            },
        );

        writer.request_flush();

        assert_eq!(woken.load(Ordering::SeqCst), 1);
    }
}
