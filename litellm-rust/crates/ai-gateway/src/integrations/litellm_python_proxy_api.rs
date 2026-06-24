//! A `CustomLogger` that ships finished events to the LiteLLM Python proxy's
//! `/v1/callbacks/logs` endpoint.
//!
//! The callback path is non-blocking: `log_success_event` / `log_failure_event`
//! build a `LogRecord` and `try_send` it onto a bounded channel, returning a
//! `LogError` (never panicking, never awaiting) if the channel is full or the
//! worker has gone away. A spawned background worker drains the channel, batches
//! records into `{"records":[...]}`, and POSTs them to the proxy with a pooled
//! `reqwest::Client`.

use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::time::interval;

use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::{
    CallbackLogsRequest, LogError, LogRecord, LoggingError, StandardLoggingPayload,
};

/// Default proxy base URL when `LITELLM_PROXY_BASE_URL` is unset.
const DEFAULT_PROXY_BASE_URL: &str = "http://localhost:4000";

/// The logs ingest path appended to the proxy base.
const CALLBACK_LOGS_PATH: &str = "/v1/callbacks/logs";

/// Bounded channel depth. Beyond this, `try_send` fails fast (we drop + count
/// rather than block the realtime splice).
const CHANNEL_CAPACITY: usize = 4096;

/// Max records flushed in one POST.
const MAX_BATCH_SIZE: usize = 256;

/// How often the worker flushes a partial batch even if it has not filled.
const FLUSH_INTERVAL: Duration = Duration::from_millis(500);

/// Ships realtime logging events to the LiteLLM Python proxy.
pub struct LiteLLMPythonProxyAPILogger {
    sink: Sender<LogRecord>,
}

impl LiteLLMPythonProxyAPILogger {
    /// Spawn the background worker and return a logger handle. `base` is the
    /// proxy base URL (no trailing path); `master_key` is sent as a bearer token.
    pub fn start(base: String, master_key: String) -> Arc<Self> {
        let (sink, receiver) = mpsc::channel::<LogRecord>(CHANNEL_CAPACITY);
        let url = format!("{}{}", base.trim_end_matches('/'), CALLBACK_LOGS_PATH);
        let client = Client::new();
        tokio::spawn(worker_loop(receiver, client, url, master_key));
        Arc::new(Self { sink })
    }

    /// Build a logger from the environment: `LITELLM_PROXY_BASE_URL` (default
    /// `http://localhost:4000`) and `LITELLM_MASTER_KEY`.
    pub fn from_env() -> Arc<Self> {
        let base = std::env::var("LITELLM_PROXY_BASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_PROXY_BASE_URL.to_string());
        let key = std::env::var("LITELLM_MASTER_KEY").unwrap_or_default();
        Self::start(base, key)
    }

    fn enqueue(&self, record: LogRecord) -> Result<(), LogError> {
        self.sink.try_send(record).map_err(|err| match err {
            mpsc::error::TrySendError::Full(_) => LogError::channel_full(),
            mpsc::error::TrySendError::Closed(_) => LogError::channel_closed(),
        })
    }
}

impl CustomLogger for LiteLLMPythonProxyAPILogger {
    fn log_success_event(&self, payload: &StandardLoggingPayload) -> Result<(), LogError> {
        self.enqueue(LogRecord {
            status: "success".to_string(),
            payload: payload.clone(),
            error: None,
        })
    }

    fn log_failure_event(
        &self,
        payload: &StandardLoggingPayload,
        error: &LoggingError,
    ) -> Result<(), LogError> {
        self.enqueue(LogRecord {
            status: "failure".to_string(),
            payload: payload.clone(),
            error: Some(format!("{}: {}", error.kind, error.message)),
        })
    }
}

/// Drain the channel, batching records and POSTing them to the proxy. Exits when
/// the channel is closed (all senders dropped) and drained.
async fn worker_loop(
    mut receiver: Receiver<LogRecord>,
    client: Client,
    url: String,
    master_key: String,
) {
    let mut ticker = interval(FLUSH_INTERVAL);
    let mut batch: Vec<LogRecord> = Vec::with_capacity(MAX_BATCH_SIZE);

    loop {
        tokio::select! {
            maybe_record = receiver.recv() => {
                match maybe_record {
                    Some(record) => {
                        batch.push(record);
                        if batch.len() >= MAX_BATCH_SIZE {
                            flush(&client, &url, &master_key, &mut batch).await;
                        }
                    }
                    None => {
                        // Channel closed: flush remaining and exit.
                        flush(&client, &url, &master_key, &mut batch).await;
                        break;
                    }
                }
            }
            _ = ticker.tick() => {
                flush(&client, &url, &master_key, &mut batch).await;
            }
        }
    }
}

/// POST the current batch (if any), clearing it. Errors are logged, not fatal.
async fn flush(client: &Client, url: &str, master_key: &str, batch: &mut Vec<LogRecord>) {
    if batch.is_empty() {
        return;
    }
    let records = std::mem::take(batch)
        .into_iter()
        .map(LogRecord::into_callback_record)
        .collect();
    let body = CallbackLogsRequest { records };

    let response = client
        .post(url)
        .bearer_auth(master_key)
        .json(&body)
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => {
            eprintln!(
                "litellm-ai-gateway: callback logs POST returned {} to {url}",
                resp.status()
            );
        }
        Err(err) => {
            eprintln!("litellm-ai-gateway: callback logs POST failed to {url}: {err}");
        }
    }
}
