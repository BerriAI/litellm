use std::time::Duration;

use serde::Serialize;

use crate::integrations::types::StandardLoggingPayload;

/// Default bounded channel depth for the log-egress worker.
/// Override: `LITELLM_LOG_CHANNEL_CAPACITY`.
const DEFAULT_CHANNEL_CAPACITY: usize = 4096;

/// Default max records POSTed per request to the control plane.
/// Override: `LITELLM_LOG_BATCH_SIZE`.
const DEFAULT_MAX_BATCH_SIZE: usize = 256;

/// Default partial-batch flush cadence, in ms.
/// Override: `LITELLM_LOG_FLUSH_INTERVAL_MS`.
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 500;

#[derive(Serialize)]
pub struct CallbackLogsRequest {
    pub records: Vec<CallbackLogRecord>,
}

#[derive(Serialize)]
pub struct CallbackLogRecord {
    pub status: String,
    pub standard_logging_payload: StandardLoggingPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LogRecord {
    pub status: String,
    pub payload: StandardLoggingPayload,
    pub error: Option<String>,
}

impl LogRecord {
    pub fn into_callback_record(self) -> CallbackLogRecord {
        CallbackLogRecord {
            status: self.status,
            standard_logging_payload: self.payload,
            error: self.error,
        }
    }
}

pub(super) struct EgressTunables {
    pub channel_capacity: usize,
    pub max_batch_size: usize,
    pub flush_interval: Duration,
}

impl EgressTunables {
    pub fn from_env() -> Self {
        Self {
            channel_capacity: env_positive(
                "LITELLM_LOG_CHANNEL_CAPACITY",
                DEFAULT_CHANNEL_CAPACITY,
            ),
            max_batch_size: env_positive("LITELLM_LOG_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE),
            flush_interval: Duration::from_millis(env_positive(
                "LITELLM_LOG_FLUSH_INTERVAL_MS",
                DEFAULT_FLUSH_INTERVAL_MS,
            )),
        }
    }
}

fn env_positive<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + From<u8>,
{
    let zero = T::from(0u8);
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<T>().ok())
        .filter(|n| *n > zero)
        .unwrap_or(default)
}
