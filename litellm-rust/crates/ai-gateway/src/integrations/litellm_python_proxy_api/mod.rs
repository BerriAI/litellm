//! A `CustomLogger` that ships finished events to the LiteLLM Python proxy's
//! `/v1/rust_control_plane/logs` endpoint.
//!
//! The callback path is non-blocking: `async_log_success_event` /
//! `async_log_failure_event`
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

use crate::constants::{DEFAULT_PROXY_BASE_URL, RUST_CONTROL_PLANE_LOGS_PATH};
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogError, LogFuture, LoggingError,
    ModelCallDetails,
};
use types::{CallbackLogsRequest, EgressTunables, LogRecord};

pub mod types;

/// Ships realtime logging events to the LiteLLM Python proxy.
pub struct LiteLLMPythonProxyAPILogger {
    sink: Sender<LogRecord>,
}

impl LiteLLMPythonProxyAPILogger {
    /// Spawn the background worker and return a logger handle. `base` is the
    /// proxy base URL (no trailing path); `master_key` is sent as a bearer token.
    pub fn start(base: String, master_key: String) -> Arc<Self> {
        let tunables = EgressTunables::from_env();
        let (sink, receiver) = mpsc::channel::<LogRecord>(tunables.channel_capacity);
        let url = format!(
            "{}{}",
            base.trim_end_matches('/'),
            RUST_CONTROL_PLANE_LOGS_PATH
        );
        let client = Client::new();
        tokio::spawn(worker_loop(
            receiver,
            client,
            url,
            master_key,
            tunables.max_batch_size,
            tunables.flush_interval,
        ));
        Arc::new(Self { sink })
    }

    /// Build a logger from the environment: `LITELLM_PROXY_BASE_URL` (default
    /// `http://localhost:4000`) and `LITELLM_MASTER_KEY`.
    ///
    /// `LITELLM_PROXY_BASE_URL` is treated as the full base and the route is
    /// appended verbatim, so if the proxy runs under a `SERVER_ROOT_PATH`
    /// (e.g. served at `https://host/litellm`), include it in the base
    /// (`LITELLM_PROXY_BASE_URL=https://host/litellm`) and the POST lands at
    /// `https://host/litellm/v1/rust_control_plane/logs`.
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
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        _response_obj: &'a CallbackValue,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if let Some(payload) = &model_call_details.standard_logging_payload {
                self.enqueue(LogRecord {
                    status: "success".to_string(),
                    payload: payload.clone(),
                    error: None,
                })?;
            }
            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        _response_obj: Option<&'a CallbackValue>,
        _timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if let Some(payload) = &model_call_details.standard_logging_payload {
                let fallback_error;
                let error = match &model_call_details.failure_error {
                    Some(error) => error,
                    None => {
                        fallback_error = LoggingError {
                            message: "callback failure event".to_string(),
                            kind: "CallbackFailure".to_string(),
                        };
                        &fallback_error
                    }
                };
                self.enqueue(LogRecord {
                    status: "failure".to_string(),
                    payload: payload.clone(),
                    error: Some(format!("{}: {}", error.kind, error.message)),
                })?;
            }
            Ok(())
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
    max_batch_size: usize,
    flush_interval: Duration,
) {
    let mut ticker = interval(flush_interval);
    let mut batch: Vec<LogRecord> = Vec::with_capacity(max_batch_size);

    loop {
        tokio::select! {
            maybe_record = receiver.recv() => {
                match maybe_record {
                    Some(record) => {
                        batch.push(record);
                        if batch.len() >= max_batch_size {
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
