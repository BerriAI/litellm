use std::sync::Arc;
use std::time::Duration;

use litellm_core::integrations::custom_guardrail::CustomGuardrail;
use litellm_core::integrations::custom_logger::CustomLogger;
use litellm_core::integrations::types::RequestMetadata;
use serde_json::{Map, Value};

pub struct AudioTranscriptionRequest<'a> {
    pub model: &'a str,
    pub audio: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub callbacks: Vec<Arc<dyn CustomLogger>>,
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub request_metadata: RequestMetadata,
    pub litellm_call_id: Option<&'a str>,
}
