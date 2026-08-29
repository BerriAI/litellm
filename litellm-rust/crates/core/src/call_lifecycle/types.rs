use std::time::Duration;

use crate::{CoreError, CoreResult};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallContext {
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
}

impl CallContext {
    pub fn new(
        model: impl Into<String>,
        custom_llm_provider: impl Into<String>,
        litellm_call_id: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            custom_llm_provider: custom_llm_provider.into(),
            litellm_call_id: litellm_call_id.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallPhase {
    BeforeCall,
    Prepare,
    BeforeSend,
    Provider,
    Complete,
}

impl CallPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BeforeCall => "before_call",
            Self::Prepare => "prepare",
            Self::BeforeSend => "before_send",
            Self::Provider => "provider",
            Self::Complete => "complete",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallPhaseTiming {
    pub phase: CallPhase,
    pub start_time: f64,
    pub end_time: f64,
    pub duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallTiming {
    pub start_time: f64,
    pub end_time: f64,
    pub phases: Vec<CallPhaseTiming>,
}

impl CallTiming {
    pub fn new(start_time: f64, end_time: f64, phases: Vec<CallPhaseTiming>) -> Self {
        Self {
            start_time,
            end_time,
            phases,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CallOutcome<'a, Response> {
    Success(&'a Response),
    Failure(&'a CoreError),
}

impl<'a, Response> CallOutcome<'a, Response> {
    pub fn from_result(result: &'a CoreResult<Response>) -> Self {
        match result {
            Ok(response) => Self::Success(response),
            Err(error) => Self::Failure(error),
        }
    }
}
