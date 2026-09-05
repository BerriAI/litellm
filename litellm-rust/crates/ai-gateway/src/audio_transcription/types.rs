use litellm_core::request_options::RequestOptions;
use std::time::Duration;

use serde_json::{Map, Value};

pub struct AudioTranscriptionRequest<'a> {
    pub model: &'a str,
    pub audio: Value,
    pub optional_params: Map<String, Value>,
    pub options: RequestOptions,
}

pub(crate) struct PreparedAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) audio: Value,
    pub(crate) provider_connection: Map<String, Value>,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}
