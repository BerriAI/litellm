use litellm_core::request_options::RequestOptions;
use std::time::Duration;

use litellm_core::ocr::transformation::OcrProviderConfig;
use serde_json::{Map, Value};

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub optional_params: Map<String, Value>,
    pub options: RequestOptions,
}

pub(crate) struct PreparedOcrRequest {
    pub(crate) config: Result<&'static dyn OcrProviderConfig, litellm_core::Error>,
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) document: Value,
    pub(crate) provider_connection: Map<String, Value>,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

pub(crate) struct ProviderOcrRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) config: &'static dyn OcrProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Value,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) timeout: Option<Duration>,
}
