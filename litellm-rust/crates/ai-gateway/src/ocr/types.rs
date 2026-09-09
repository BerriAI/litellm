use crate::integrations::custom_guardrail::CustomGuardrail;
use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::RequestMetadata;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::Duration;

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
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
