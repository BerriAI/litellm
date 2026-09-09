use std::sync::Arc;
use std::time::Duration;

use litellm_core::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogError, LogFuture, LoggingError,
    ModelCallDetails,
};
use litellm_core::integrations::types::StandardLoggingPayload;
use reqwest::Client;
use serde::Serialize;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::time::interval;

const RUST_CONTROL_PLANE_LOGS_PATH: &str = "/v1/rust_control_plane/logs";

pub(crate) struct LiteLLMPythonProxyAPILogger {
    sink: Sender<LogRecord>,
}

impl LiteLLMPythonProxyAPILogger {
    pub(crate) fn start(base: String, master_key: String, config: LogEgressConfig) -> Arc<Self> {
        let (sink, receiver) = mpsc::channel(config.channel_capacity);
        let url = format!(
            "{}{}",
            base.trim_end_matches('/'),
            RUST_CONTROL_PLANE_LOGS_PATH
        );
        tokio::spawn(worker_loop(
            receiver,
            Client::new(),
            url,
            master_key,
            config.max_batch_size,
            config.flush_interval,
        ));
        Arc::new(Self { sink })
    }

    fn enqueue(&self, record: LogRecord) -> Result<(), LogError> {
        self.sink.try_send(record).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => LogError::channel_full(),
            mpsc::error::TrySendError::Closed(_) => LogError::channel_closed(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LogEgressConfig {
    pub channel_capacity: usize,
    pub max_batch_size: usize,
    pub flush_interval: Duration,
}

impl CustomLogger for LiteLLMPythonProxyAPILogger {
    fn async_log_success_event<'a>(
        &'a self,
        details: &'a ModelCallDetails,
        _: &'a CallbackValue,
        _: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if let Some(payload) = &details.standard_logging_payload {
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
        details: &'a ModelCallDetails,
        _: Option<&'a CallbackValue>,
        _: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            if let Some(payload) = &details.standard_logging_payload {
                let fallback;
                let error = match &details.failure_error {
                    Some(error) => error,
                    None => {
                        fallback = LoggingError {
                            message: "callback failure event".to_string(),
                            kind: "CallbackFailure".to_string(),
                        };
                        &fallback
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

#[derive(Clone, Debug)]
struct LogRecord {
    status: String,
    payload: StandardLoggingPayload,
    error: Option<String>,
}

#[derive(Serialize)]
struct CallbackLogsRequest {
    records: Vec<CallbackLogRecord>,
}

#[derive(Serialize)]
struct CallbackLogRecord {
    status: String,
    standard_logging_payload: StandardLoggingPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn worker_loop(
    mut receiver: Receiver<LogRecord>,
    client: Client,
    url: String,
    master_key: String,
    max_batch_size: usize,
    flush_interval: Duration,
) {
    let mut ticker = interval(flush_interval);
    let mut batch = Vec::with_capacity(max_batch_size);
    loop {
        tokio::select! {
            record = receiver.recv() => match record {
                Some(record) => {
                    batch.push(record);
                    if batch.len() >= max_batch_size {
                        flush(&client, &url, &master_key, &mut batch).await;
                    }
                }
                None => {
                    flush(&client, &url, &master_key, &mut batch).await;
                    break;
                }
            },
            _ = ticker.tick() => flush(&client, &url, &master_key, &mut batch).await,
        }
    }
}

async fn flush(client: &Client, url: &str, master_key: &str, batch: &mut Vec<LogRecord>) {
    if batch.is_empty() {
        return;
    }
    let records = std::mem::take(batch)
        .into_iter()
        .map(|record| CallbackLogRecord {
            status: record.status,
            standard_logging_payload: record.payload,
            error: record.error,
        })
        .collect();
    let response = client
        .post(url)
        .bearer_auth(master_key)
        .json(&CallbackLogsRequest { records })
        .send()
        .await;
    match response {
        Ok(response) if response.status().is_success() => {}
        Ok(response) => eprintln!(
            "litellm-ai-gateway: callback logs POST returned {} to {url}",
            response.status()
        ),
        Err(error) => eprintln!("litellm-ai-gateway: callback logs POST failed to {url}: {error}"),
    }
}
