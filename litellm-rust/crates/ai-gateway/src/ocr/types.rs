use std::sync::Arc;
use std::time::Duration;

use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleRequest};
use litellm_core::ocr::transformation::OcrProviderConfig;
use serde_json::{Map, Value};

use crate::integrations::custom_guardrail::CustomGuardrail;
use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::RequestMetadata;

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

pub(crate) struct PreparedOcrRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) litellm_call_id: String,
    pub(crate) document: Value,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

impl CallLifecycleRequest for PreparedOcrRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext {
        CallLifecycleContext::new(
            "ocr",
            self.model.clone(),
            self.custom_llm_provider.clone(),
            self.litellm_call_id.clone(),
        )
    }
}

pub(crate) struct ProviderOcrRequest {
    pub(crate) model: String,
    pub(crate) config: &'static dyn OcrProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Value,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) timeout: Option<Duration>,
}
