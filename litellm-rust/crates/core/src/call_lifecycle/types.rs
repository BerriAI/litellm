use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallLifecycleContext {
    pub call_type: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
}

impl CallLifecycleContext {
    pub fn new(
        call_type: impl Into<String>,
        model: impl Into<String>,
        custom_llm_provider: impl Into<String>,
        litellm_call_id: impl Into<String>,
    ) -> Self {
        Self {
            call_type: call_type.into(),
            model: model.into(),
            custom_llm_provider: custom_llm_provider.into(),
            litellm_call_id: litellm_call_id.into(),
        }
    }
}

pub trait CallLifecycleRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallLifecyclePhase {
    PreCall,
    DuringCall,
    ProviderCall,
    SuccessCallback,
    FailureCallback,
}

impl CallLifecyclePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PreCall => "pre_call",
            Self::DuringCall => "during_call",
            Self::ProviderCall => "provider_call",
            Self::SuccessCallback => "success_callback",
            Self::FailureCallback => "failure_callback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallLifecyclePhaseTiming {
    pub phase: CallLifecyclePhase,
    pub start_time: f64,
    pub end_time: f64,
    pub duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallLifecycleTiming {
    pub start_time: f64,
    pub end_time: f64,
    pub phases: Vec<CallLifecyclePhaseTiming>,
}

impl CallLifecycleTiming {
    pub fn new(start_time: f64, end_time: f64, phases: Vec<CallLifecyclePhaseTiming>) -> Self {
        Self {
            start_time,
            end_time,
            phases,
        }
    }
}
