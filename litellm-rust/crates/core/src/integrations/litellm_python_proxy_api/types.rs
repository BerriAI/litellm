use serde::Serialize;

use crate::integrations::types::StandardLoggingPayload;

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
