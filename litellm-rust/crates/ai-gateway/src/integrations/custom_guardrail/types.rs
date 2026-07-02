use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::integrations::custom_logger::CallType;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardrailError {
    pub message: String,
    pub kind: String,
}

impl GuardrailError {
    pub fn blocked(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: "GuardrailBlocked".to_string(),
        }
    }
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for GuardrailError {}

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
