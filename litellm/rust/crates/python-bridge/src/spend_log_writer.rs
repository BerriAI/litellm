//! Python binding for the Rust spend log writer (`litellm-spend-log-writer`).
//!
//! The proxy installs one writer per Prisma client on the ledger-on path. Python hands rows over
//! once, as `json.dumps` text, and its spend log monitor calls `flush` for each pass, so rows are
//! written straight from the Rust queue. Rows come back to Python only to be parked in Redis.

use std::sync::Arc;

use litellm_host_python::release_gil;
use litellm_repositories::SpendLogRepository;
use litellm_spend_log_writer::{
    PgLedgerStore, QueuedSpendLog, Retention, SpendLogFlusher, SpendLogWriter, WriterSettings,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::db::NativeDatabase;

type RetentionFields = (Option<Vec<String>>, Option<Vec<String>>);

/// The fields of a written log that `process_spend_logs_guardrail_usage` reads, for a log whose
/// metadata carries guardrail results.
#[derive(Serialize)]
struct GuardrailUsageRow {
    request_id: String,
    #[serde(rename = "startTime")]
    start_time: String,
    team_id: String,
    api_key: String,
    metadata: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    accounting_payload: Option<Map<String, Value>>,
}

fn guardrail_usage_rows(rows: &[QueuedSpendLog]) -> Vec<GuardrailUsageRow> {
    rows.iter()
        .filter_map(|entry| {
            let row = entry.row()?;
            row.metadata
                .contains("guardrail_information")
                .then(|| GuardrailUsageRow {
                    request_id: row.request_id.clone(),
                    start_time: row.start_time.clone(),
                    team_id: row.team_id.clone(),
                    api_key: row.api_key.clone(),
                    metadata: row.metadata.clone(),
                    accounting_payload: entry.charge().map(|charge| charge.payload.clone()),
                })
        })
        .collect()
}

enum PassError {
    Database(litellm_db::Error),
    Aborted(String),
}

fn map_pass_error(error: PassError) -> PyErr {
    match error {
        PassError::Database(error) => crate::db::map_error(error),
        PassError::Aborted(reason) => PyRuntimeError::new_err(reason),
    }
}

/// The Rust spend log queue and writer the proxy uses on the ledger-on path.
#[pyclass(frozen, name = "RustSpendLogWriter")]
pub struct RustSpendLogWriter {
    writer: Arc<SpendLogWriter<PgLedgerStore>>,
}

impl RustSpendLogWriter {
    /// The installed writer, for the Rust spend logger that enqueues rows into it.
    pub(crate) fn shared(&self) -> Arc<SpendLogWriter<PgLedgerStore>> {
        Arc::clone(&self.writer)
    }
}

#[pymethods]
impl RustSpendLogWriter {
    /// `on_flush_request` is called, with the GIL, whenever a row asks to be written at once; the
    /// proxy passes a callable that wakes its spend log monitor from any thread.
    #[new]
    #[pyo3(signature = (database, *, queue_max_bytes, write_batch_max_bytes, write_batch_max_rows, on_flush_request))]
    fn new(
        database: PyRef<'_, NativeDatabase>,
        queue_max_bytes: usize,
        write_batch_max_bytes: usize,
        write_batch_max_rows: usize,
        on_flush_request: Py<PyAny>,
    ) -> PyResult<Self> {
        let settings = WriterSettings {
            queue_max_bytes,
            write_batch_max_bytes,
            write_batch_max_rows,
            ..WriterSettings::default()
        };
        let store = PgLedgerStore::new(
            SpendLogRepository::new(database.repository_database()?),
            settings.transaction_timeout,
        );
        let wake = move || {
            Python::attach(|py| {
                if let Err(error) = on_flush_request.call0(py) {
                    tracing::warn!(
                        "Spend tracking - could not request a spend log flush: {}",
                        error
                    );
                }
            });
        };
        Ok(Self {
            writer: Arc::new(SpendLogWriter::new(
                SpendLogFlusher::new(store, settings),
                wake,
            )),
        })
    }

    /// Queues rows Python built, each as its `json.dumps` text, within `max_bytes`. A row the
    /// writer cannot hold is logged and dropped, keeping its charge. Returns how many of the
    /// oldest queued rows were dropped to stay within the budget.
    fn enqueue(&self, py: Python<'_>, rows: Vec<String>, at_head: bool, max_bytes: usize) -> usize {
        let logger = crate::logger::capture(py);
        let writer = Arc::clone(&self.writer);
        release_gil(py, move || {
            logger.scope(|| {
                let queued = rows
                    .iter()
                    .filter_map(|text| match QueuedSpendLog::from_python(text) {
                        Ok(entry) => Some(entry),
                        Err(rejected) => {
                            tracing::error!(
                                "Spend tracking - dropping malformed spend log row. request_id={} error={}",
                                rejected.request_id.as_deref().unwrap_or("None"),
                                rejected.error
                            );
                            rejected.charge.map(|charge| *charge)
                        }
                    })
                    .collect();
                writer.enqueue_within(queued, at_head, max_bytes)
            })
        })
    }

    /// Writes one pass of up to `limit` rows from the head of the queue, cutting each logged row's
    /// metadata to `retention` (`include`, `exclude`). The pass runs to the end even when the
    /// awaiting task is cancelled, and a failed pass requeues its rows at the head before it
    /// raises `NativeDatabaseFailure`. Returns the written logs the guardrail usage rollup reads.
    #[pyo3(signature = (limit, retention=None))]
    fn flush<'py>(
        &self,
        py: Python<'py>,
        limit: usize,
        retention: Option<RetentionFields>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let retention = retention.map(|(include, exclude)| Retention { include, exclude });
        let writer = Arc::clone(&self.writer);
        let pass = Box::pin(crate::db::for_caller(
            py,
            crate::logger::capture(py).instrument(async move {
                Box::pin(writer.flush(limit, retention.as_ref()))
                    .await
                    .map(|rows| guardrail_usage_rows(&rows))
            }),
        )?);
        crate::execution::run_async(
            py,
            async move {
                match litellm_call_scope::spawn_detached(pass).await {
                    Ok(result) => result.map_err(PassError::Database),
                    Err(error) => Err(PassError::Aborted(error.to_string())),
                }
            },
            map_pass_error,
        )
    }

    /// Waits until no pass is writing, so rows a pass whose caller was cancelled puts back are in
    /// the queue before shutdown drains or parks it.
    fn wait_idle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let writer = Arc::clone(&self.writer);
        crate::execution::run_async(
            py,
            async move {
                writer.idle().await;
                Ok::<(), PassError>(())
            },
            map_pass_error,
        )
    }

    /// Takes up to `limit` rows off the head without writing them, as the JSON objects Python's
    /// queue holds, for the proxy to park in Redis.
    fn dequeue(&self, limit: usize) -> Vec<String> {
        self.writer
            .take(limit)
            .iter()
            .map(QueuedSpendLog::to_json)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.writer.len()
    }
}
