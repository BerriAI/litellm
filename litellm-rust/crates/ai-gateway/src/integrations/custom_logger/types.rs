use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};

pub type LogFuture<'a> = Pin<Box<dyn Future<Output = Result<(), LogError>> + Send + 'a>>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallbackDispatchReport {
    pub invoked: usize,
    pub dropped: usize,
}

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
        self.model = payload.model.clone();
        self.custom_llm_provider = payload.custom_llm_provider.clone();
        self.call_type = CallType::from(payload.call_type.as_str());
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

#[derive(Clone, Debug)]
pub struct LoggingError {
    pub message: String,
    pub kind: String,
}

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
