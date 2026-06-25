//! Typed payloads for the LiteLLM `/v1/callbacks/logs` realtime-logging contract.
//!
//! Field names below are the EXACT JSON keys the Python replay path + spend-logs
//! builder read. Note the deliberate mix:
//!   - `startTime` / `endTime` are camelCase (epoch f64 seconds)
//!   - `response_cost` / `prompt_tokens` / etc. are snake_case
//!
//! Mirrors Python `litellm/integrations/` + the proxy `CallbackLogsRequest`
//! contract 1:1.

use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

/// Cumulative token usage for a realtime session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Cost-attribution metadata threaded from the authenticated request.
#[derive(Clone, Debug, Default)]
pub struct RequestMetadata {
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
}

/// LiteLLM call type carried through callback and guardrail hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallType {
    Ocr,
    Realtime,
    Completion,
    Acompletion,
    ChatCompletion,
    Other(String),
}

impl CallType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Ocr => "ocr",
            Self::Realtime => "realtime",
            Self::Completion => "completion",
            Self::Acompletion => "acompletion",
            Self::ChatCompletion => "chat_completion",
            Self::Other(value) => value.as_str(),
        }
    }
}

impl From<&str> for CallType {
    fn from(value: &str) -> Self {
        match value {
            "ocr" => Self::Ocr,
            "realtime" => Self::Realtime,
            "completion" => Self::Completion,
            "acompletion" => Self::Acompletion,
            "chat_completion" => Self::ChatCompletion,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for CallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Python-compatible callback timing, kept typed in Rust.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackTiming {
    pub start_time: f64,
    pub end_time: f64,
}

impl CallbackTiming {
    pub fn new(start_time: f64, end_time: f64) -> Self {
        Self {
            start_time,
            end_time,
        }
    }
}

/// Response or error object passed to a callback. The `object` field mirrors the
/// Python response object's `.object` convention where available.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackValue {
    pub object: String,
    pub value: Value,
}

impl CallbackValue {
    pub fn new(object: impl Into<String>, value: Value) -> Self {
        Self {
            object: object.into(),
            value,
        }
    }
}

/// Typed equivalent of Python `model_call_details` for terminal callback hooks.
#[derive(Clone, Debug)]
pub struct ModelCallDetails {
    pub model: String,
    pub custom_llm_provider: String,
    pub call_type: CallType,
    pub metadata: StandardLoggingMetadata,
    pub extra_metadata: HashMap<String, Value>,
    pub request_id: Option<String>,
    pub litellm_call_id: Option<String>,
    pub response_cost: Option<f64>,
    pub standard_logging_payload: Option<StandardLoggingPayload>,
    pub failure_error: Option<LoggingError>,
}

impl ModelCallDetails {
    pub fn new(
        model: impl Into<String>,
        custom_llm_provider: impl Into<String>,
        call_type: CallType,
    ) -> Self {
        Self {
            model: model.into(),
            custom_llm_provider: custom_llm_provider.into(),
            call_type,
            metadata: StandardLoggingMetadata::default(),
            extra_metadata: HashMap::new(),
            request_id: None,
            litellm_call_id: None,
            response_cost: None,
            standard_logging_payload: None,
            failure_error: None,
        }
    }

    pub fn from_standard_logging_payload(payload: StandardLoggingPayload) -> Self {
        let request_id = Some(payload.id.clone());
        let litellm_call_id = Some(payload.litellm_call_id.clone());
        let response_cost = Some(payload.response_cost);
        let metadata = payload.metadata.clone();
        Self {
            model: payload.model.clone(),
            custom_llm_provider: payload.custom_llm_provider.clone(),
            call_type: CallType::from(payload.call_type.as_str()),
            metadata,
            extra_metadata: HashMap::new(),
            request_id,
            litellm_call_id,
            response_cost,
            standard_logging_payload: Some(payload),
            failure_error: None,
        }
    }

    pub fn with_standard_logging_payload(mut self, payload: StandardLoggingPayload) -> Self {
        self.request_id = Some(payload.id.clone());
        self.litellm_call_id = Some(payload.litellm_call_id.clone());
        self.response_cost = Some(payload.response_cost);
        self.metadata = payload.metadata.clone();
        self.standard_logging_payload = Some(payload);
        self
    }

    pub fn with_failure_error(mut self, error: LoggingError) -> Self {
        self.failure_error = Some(error);
        self
    }
}

/// A logging-callback failure (e.g. a custom logger raised). Mirrors the Python
/// failure-event shape: a message plus an exception kind/class name.
#[derive(Clone, Debug)]
pub struct LoggingError {
    pub message: String,
    pub kind: String,
}

/// A non-fatal error returned by a `CustomLogger` when it cannot enqueue an
/// event (channel full or the background worker has shut down).
#[derive(Clone, Debug)]
pub struct LogError {
    pub message: String,
    pub kind: String,
}

impl LogError {
    pub fn channel_full() -> Self {
        Self {
            message: "logging channel is full; dropping record".to_string(),
            kind: "ChannelFull".to_string(),
        }
    }

    pub fn channel_closed() -> Self {
        Self {
            message: "logging channel is closed; worker has shut down".to_string(),
            kind: "ChannelClosed".to_string(),
        }
    }
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for LogError {}

/// Batch wrapper — the top-level request body.
/// Matches Python `CallbackLogsRequest { records: list[CallbackLogRecord] }`.
#[derive(Serialize)]
pub struct CallbackLogsRequest {
    pub records: Vec<CallbackLogRecord>,
}

/// One finished logging event.
/// Matches `CallbackLogRecord { status, standard_logging_payload, error? }`.
#[derive(Serialize)]
pub struct CallbackLogRecord {
    /// "success" | "failure". On "failure", `error` (or payload.error_str)
    /// becomes the replayed exception string.
    pub status: String,

    pub standard_logging_payload: StandardLoggingPayload,

    /// Only meaningful when status == "failure". Omitted on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The self-describing payload. Field names are the EXACT JSON keys the Python
/// replay path + spend-logs builder read.
#[derive(Clone, Debug, Serialize)]
pub struct StandardLoggingPayload {
    pub id: String,
    pub litellm_call_id: String,

    /// e.g. "realtime", "acompletion". Falls back to "acompletion" if absent.
    pub call_type: String,

    pub model: String,
    pub custom_llm_provider: String,

    /// Spend ($) written to LiteLLM_SpendLogs.spend.
    pub response_cost: f64,

    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,

    /// EPOCH SECONDS as float — camelCase keys, NOT snake_case.
    #[serde(rename = "startTime")]
    pub start_time: f64,
    #[serde(rename = "endTime")]
    pub end_time: f64,

    pub stream: bool,

    pub metadata: StandardLoggingMetadata,

    /// Optional; stored as request input on the spend log row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Value>,
}

/// Cost-attribution keys. The replayer maps these into litellm_params.metadata,
/// which the spend-logs builder reads to set user / team_id / organization_id.
#[derive(Clone, Debug, Serialize, Default)]
pub struct StandardLoggingMetadata {
    pub user_api_key_hash: Option<String>, // -> SpendLogs.api_key
    pub user_api_key_user_id: Option<String>, // -> SpendLogs.user
    pub user_api_key_team_id: Option<String>, // -> SpendLogs.team_id

    // Optional but read by the builder; include when known:
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_alias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_org_id: Option<String>, // -> SpendLogs.organization_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_api_key_end_user_id: Option<String>, // -> SpendLogs.end_user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spend_logs_metadata: Option<HashMap<String, Value>>,
}

/// The unit handed to a `CustomLogger` sink: a finished payload plus its status
/// and (on failure) the replayed error string.
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
