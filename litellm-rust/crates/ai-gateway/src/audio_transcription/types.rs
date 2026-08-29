use std::time::Duration;

use litellm_core::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use litellm_core::call_lifecycle::CallSpec;
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
    pub litellm_call_id: Option<&'a str>,
}

pub enum AudioTranscriptionCall {}

impl CallSpec for AudioTranscriptionCall {
    const NAME: &'static str = "audio_transcription";
    type BeforeCall = PreparedAudioTranscriptionRequest;
    type BeforeSend = ProviderAudioTranscriptionRequest;
    type Response = Value;
}

pub struct PreparedAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub audio: Value,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub(crate) private_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

#[derive(Clone)]
pub struct ProviderAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) config: &'static dyn AudioTranscriptionProviderConfig,
    pub(crate) url: String,
    pub body: Value,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

impl ProviderAudioTranscriptionRequest {
    pub fn endpoint(&self) -> &str {
        &self.url
    }
}
