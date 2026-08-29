use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use thiserror::Error;

use crate::callbacks::custom_logger::CallType;
use crate::error::{CoreError, RequestError};

pub type GuardrailFuture<'a> =
    Pin<Box<dyn Future<Output = Result<GuardrailDecision, GuardrailError>> + Send + 'a>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardrailEventHook {
    PreCall,
    DuringCall,
}

impl GuardrailEventHook {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreCall => "pre_call",
            Self::DuringCall => "during_call",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum GuardrailError {
    /// A guardrail policy rejected the request or response content.
    #[error("GuardrailBlocked: {0}")]
    Blocked(String),
}

impl GuardrailError {
    pub fn blocked(message: impl Into<String>) -> Self {
        Self::Blocked(message.into())
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Blocked(_) => "GuardrailBlocked",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Blocked(message) => message,
        }
    }
}

/// A guardrail can only run before the provider call is issued, so its
/// failures are always request-side: the host may retry on its own path.
impl From<GuardrailError> for CoreError {
    fn from(error: GuardrailError) -> Self {
        Self::Request(RequestError::InvalidRequest(error.to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct GuardrailContext {
    pub call_type: CallType,
    pub selected_guardrails: Vec<String>,
    pub metadata: HashMap<String, Value>,
    pub user_api_key_hash: Option<String>,
    pub user_api_key_user_id: Option<String>,
    pub user_api_key_team_id: Option<String>,
    pub trace_parent: Option<String>,
}

impl GuardrailContext {
    pub fn new(call_type: CallType) -> Self {
        Self {
            call_type,
            selected_guardrails: Vec::new(),
            metadata: HashMap::new(),
            user_api_key_hash: None,
            user_api_key_user_id: None,
            user_api_key_team_id: None,
            trace_parent: None,
        }
    }

    pub fn with_selected_guardrails(mut self, selected_guardrails: Vec<String>) -> Self {
        self.selected_guardrails = selected_guardrails;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GuardrailRequest {
    pub data: Value,
}

impl GuardrailRequest {
    pub fn new(data: Value) -> Self {
        Self { data }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GuardrailDecision {
    Allow(GuardrailRequest),
    Mask(GuardrailRequest),
    Block(GuardrailError),
}

impl GuardrailDecision {
    pub(super) fn into_request(self) -> Result<GuardrailRequest, GuardrailError> {
        match self {
            Self::Allow(request) | Self::Mask(request) => Ok(request),
            Self::Block(error) => Err(error),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuardrailDispatchReport {
    pub invoked: usize,
}
