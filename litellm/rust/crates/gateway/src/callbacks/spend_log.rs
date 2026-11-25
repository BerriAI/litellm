//! The proxy's spend log callback (`_ProxyDBLogger`) over the engine in
//! `litellm-callbacks-spend-logger`, writing each served call's row to the gateway's database,
//! with `RustSpendLogHost` answering its host calls.
//!
//! The Python proxy's spend log monitor decides when the writer flushes, so the gateway runs its
//! own: a pass every `PROXY_BATCH_WRITE_AT` seconds and its jitter, or at once when a row asks for
//! it, and a last drain at shutdown. A pass that fails keeps its rows at the head of the queue for
//! the next one. Python parks a failed pass in its Redis transaction buffer, which only a config
//! `admit` leaves to Python turns on.

use std::{
    sync::{Arc, Weak},
    time::Duration,
};

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_config::Environment;
use litellm_redaction::TokenRedactor;
use litellm_redis_store::{Remote, spend_counters::SpendCounters};
use litellm_repositories::{Database, SpendLogRepository};
use litellm_spend_log_writer::{
    LedgerStore, PgLedgerStore, SpendLogFlusher, SpendLogWriter, WriterSettings,
};
use rand::Rng;
use tokio::sync::Notify;

use super::{
    Registered, budget_reservation::Reserver, parallel_request_limiter::REDIS_DEFAULT_TTL,
};
use crate::{
    RedisClients,
    host::{Rows, RustSpendLogHost},
};

/// `PROXY_BATCH_WRITE_AT` when unset.
const BATCH_WRITE_AT: Duration = Duration::from_secs(10);
/// `_BATCH_WRITE_JITTER_MAX_SECONDS`.
const JITTER_MAX_SECONDS: u64 = 5;
/// `MAX_LOGS_PER_INTERVAL`.
const PASS_ROWS: usize = 10_000;
/// How long shutdown waits for the queue to drain before it gives the rest up.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

fn settings(environment: &Environment) -> WriterSettings {
    let defaults = WriterSettings::default();
    let at_least_one = |set: Option<u64>, default: usize| {
        set.map_or(default, |set| {
            usize::try_from(set).unwrap_or(usize::MAX).max(1)
        })
    };
    WriterSettings {
        queue_max_bytes: at_least_one(
            environment.spend_log_queue_max_bytes,
            defaults.queue_max_bytes,
        ),
        write_batch_max_bytes: at_least_one(
            environment.spend_log_write_batch_max_bytes,
            defaults.write_batch_max_bytes,
        ),
        write_batch_max_rows: at_least_one(
            environment.spend_log_write_batch_max_rows,
            defaults.write_batch_max_rows,
        ),
        ..defaults
    }
}

/// The spend log writer the callback queues to, and what wakes its flush loop.
pub struct SpendLogs<S: LedgerStore = PgLedgerStore> {
    writer: Arc<SpendLogWriter<S>>,
    wake: Arc<Notify>,
    interval: Duration,
}

/// The spend log callback and what it keeps for the life of the process: the writer it queues
/// to, the spend counters it charges and the host that answers it. A reloaded config registers the
/// same callback again, so no row queued under the old config is lost.
pub struct SpendLog {
    logs: Arc<SpendLogs>,
    callback: Arc<dyn ErasedCallback>,
    host: Arc<RustSpendLogHost>,
    /// The database rows a budget reservation's counters are seeded and floored from.
    rows: Arc<Rows>,
    /// The spend counters the callback charges, which a budget reservation holds against.
    budgets: Arc<Budgets>,
}

impl SpendLog {
    /// The callback over `database`, charging spend counters in `redis`'s coordination client
    /// when there is one, and the writer it queues to, whose flush loop runs from here on.
    pub fn start(database: &Database, redis: &RedisClients, environment: &Environment) -> Self {
        let settings = settings(environment);
        let store = PgLedgerStore::new(
            SpendLogRepository::new(database.clone()),
            settings.transaction_timeout,
        );
        let interval = environment
            .proxy_batch_write_at
            .map_or(BATCH_WRITE_AT, Duration::from_secs);
        let logs = SpendLogs::flushed(store, settings, interval);
        let counters = SpendCounters::new(redis.coordination.clone().map(|client| Remote {
            client,
            default_ttl: REDIS_DEFAULT_TTL,
        }));
        let budgets = Arc::new(Budgets::new(Arc::new(counters)));
        let callback = SpendLogCallback::new(SpendLogParts {
            writer: logs.writer.clone(),
            budgets: budgets.clone(),
            tokens: Arc::new(TokenRedactor::new(0, Vec::new())),
        });
        let rows = Arc::new(Rows::new(database.clone()));
        Self {
            logs,
            callback: Arc::new(callback),
            host: Arc::new(RustSpendLogHost::new(Some(rows.clone()))),
            rows,
            budgets,
        }
    }

    /// The budget reservation over the counters this callback charges, so what it holds is what
    /// the callback settles, with a cold counter started from its row's spend.
    pub(crate) fn reserver(&self) -> Arc<Reserver> {
        Reserver::new(self.budgets.clone(), self.rows.clone())
    }

    /// The spend counters the callback charges and a reservation holds against, which a spend
    /// reset another pod announces is applied to.
    pub(crate) fn spend_counters(&self) -> &Arc<SpendCounters> {
        self.budgets.store()
    }

    /// The callback as one config's callback stage runs it.
    pub(crate) fn registered(&self) -> Registered {
        Registered::hosted(self.callback.clone(), self.host.clone())
    }

    /// Writes what is still queued, for up to [`DRAIN_DEADLINE`], once the gateway stops serving.
    pub async fn drain(&self) {
        self.logs.drain().await;
    }
}

impl<S: LedgerStore + 'static> SpendLogs<S> {
    /// A writer over `store` whose flush loop runs every `interval` from here on.
    fn flushed(store: S, settings: WriterSettings, interval: Duration) -> Arc<Self> {
        let wake = Arc::new(Notify::new());
        let woken = wake.clone();
        let writer = Arc::new(SpendLogWriter::new(
            SpendLogFlusher::new(store, settings),
            move || woken.notify_one(),
        ));
        let logs = Arc::new(Self {
            writer,
            wake,
            interval,
        });
        tokio::spawn(spend_logs_write_loop(Arc::downgrade(&logs)));
        logs
    }

    /// Writes batches until the queue is empty or a batch write fails.
    async fn flush(&self) {
        while !self.writer.is_empty() {
            if let Err(error) = Box::pin(self.writer.flush(PASS_ROWS, None)).await {
                tracing::error!(
                    error = %litellm_error::Report::new(&error),
                    queued = self.writer.len(),
                    "Spend tracking - the spend log pass failed, its rows stay queued"
                );
                return;
            }
        }
    }

    /// Writes what is still queued, for up to [`DRAIN_DEADLINE`], once the gateway stops serving.
    pub async fn drain(&self) {
        if tokio::time::timeout(DRAIN_DEADLINE, async {
            self.writer.idle().await;
            self.flush().await;
        })
        .await
        .is_err()
        {
            tracing::error!(
                queued = self.writer.len(),
                "Spend tracking - shutdown gave up on the spend logs still queued"
            );
        }
    }
}

/// The flush loop of `logs`, until the gateway drops them.
async fn spend_logs_write_loop<S: LedgerStore + 'static>(logs: Weak<SpendLogs<S>>) {
    loop {
        let Some((wake, interval)) = logs
            .upgrade() // this fails when the process drops the spendlogs
            .map(|logs| (logs.wake.clone(), logs.interval))
        else {
            return;
        };
        let jitter = Duration::from_secs(rand::thread_rng().gen_range(0..=JITTER_MAX_SECONDS));
        tokio::select! {
            () = tokio::time::sleep(interval + jitter) => {}
            () = wake.notified() => {}
        }
        let Some(logs) = logs.upgrade() else {
            return;
        };
        logs.flush().await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use litellm_repositories::SpendLogWrite;
    use litellm_spend_log_writer::QueuedSpendLog;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    /// A ledger that counts the rows it wrote, or refuses every write while `down`.
    #[derive(Default)]
    struct Ledger {
        written: AtomicUsize,
        down: AtomicBool,
    }

    struct Shared(Arc<Ledger>);

    impl LedgerStore for Shared {
        async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, litellm_db::Error> {
            if self.0.down.load(Ordering::SeqCst) {
                return Err(litellm_db::Error::TransactionTimedOut);
            }
            self.0.written.fetch_add(writes.len(), Ordering::SeqCst);
            Ok(writes.len())
        }
    }

    fn row(request_id: &str) -> QueuedSpendLog {
        let row = json!({
            "request_id": request_id,
            "call_type": "anthropic_messages",
            "api_key": "hashed",
            "spend": 0.25,
            "total_tokens": 3,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "startTime": "2026-11-02T12:00:00+00:00",
            "endTime": "2026-11-02T12:00:01+00:00",
            "completionStartTime": null,
            "model": "claude",
            "model_id": "deployment",
            "model_group": "claude",
            "mcp_namespaced_tool_name": null,
            "agent_id": null,
            "billing_agent_id": null,
            "api_base": "https://example.test",
            "user": "user",
            "metadata": "{\"status\": \"success\"}",
            "cache_hit": "False",
            "cache_key": "Cache OFF",
            "request_tags": "[]",
            "team_id": "",
            "organization_id": "",
            "end_user": "",
            "requester_ip_address": null,
            "custom_llm_provider": "anthropic",
            "messages": "[]",
            "response": "{}",
            "proxy_server_request": "{}",
            "session_id": null,
            "request_duration_ms": 1000,
            "status": "success",
            "litellm_call_id": null,
        });
        QueuedSpendLog::from_python(&row.to_string()).unwrap()
    }

    /// Spend logs over `ledger` with `queued` rows waiting and a flush loop that only a wake
    /// starts.
    fn queued(ledger: &Arc<Ledger>, queued: usize) -> Arc<SpendLogs<Shared>> {
        let logs = SpendLogs::flushed(
            Shared(ledger.clone()),
            WriterSettings::default(),
            Duration::from_secs(3_600),
        );
        let rows = (0..queued).map(|n| row(&format!("row-{n}"))).collect();
        logs.writer.enqueue(rows, false);
        logs
    }

    async fn written(ledger: &Ledger, rows: usize) -> usize {
        for _ in 0..200 {
            if ledger.written.load(Ordering::SeqCst) >= rows {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        ledger.written.load(Ordering::SeqCst)
    }

    #[rstest]
    #[tokio::test]
    async fn a_row_that_asks_for_a_pass_is_written_before_the_interval() {
        let ledger = Arc::new(Ledger::default());
        let logs = queued(&ledger, 3);
        logs.writer.request_flush();
        assert_eq!((written(&ledger, 3).await, logs.writer.len()), (3, 0));
    }

    #[rstest]
    #[tokio::test]
    async fn a_pass_that_fails_keeps_its_rows_for_the_next() {
        let ledger = Arc::new(Ledger::default());
        ledger.down.store(true, Ordering::SeqCst);
        let logs = queued(&ledger, 2);
        logs.flush().await;
        let kept = logs.writer.len();
        ledger.down.store(false, Ordering::SeqCst);
        logs.flush().await;
        assert_eq!(
            (
                kept,
                ledger.written.load(Ordering::SeqCst),
                logs.writer.len()
            ),
            (2, 2, 0)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn shutdown_writes_what_is_still_queued() {
        let ledger = Arc::new(Ledger::default());
        let logs = queued(&ledger, 4);
        logs.drain().await;
        assert_eq!(
            (ledger.written.load(Ordering::SeqCst), logs.writer.len()),
            (4, 0)
        );
    }

    #[rstest]
    #[case::the_proxys_defaults(Environment::default(), (64_000_000, 2_000_000, 100))]
    #[case::the_variables_set(
        Environment {
            spend_log_queue_max_bytes: Some(1_000),
            spend_log_write_batch_max_bytes: Some(500),
            spend_log_write_batch_max_rows: Some(0),
            ..Environment::default()
        },
        (1_000, 500, 1)
    )]
    fn the_writers_limits_come_from_the_environment(
        #[case] environment: Environment,
        #[case] limits: (usize, usize, usize),
    ) {
        let settings = settings(&environment);
        assert_eq!(
            (
                settings.queue_max_bytes,
                settings.write_batch_max_bytes,
                settings.write_batch_max_rows
            ),
            limits
        );
    }
}
