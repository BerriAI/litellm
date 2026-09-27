use std::time::Duration;

use litellm_llms::base_llm::{
    audio_transcription::transformation::BaseAudioTranscriptionConfig, auth::ValidatedEnvironment,
};
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
}

#[derive(Clone)]
pub struct ProviderAudioTranscriptionRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub config: &'static dyn BaseAudioTranscriptionConfig,
    pub url: String,
    pub body: Value,
    pub environment: ValidatedEnvironment,
    pub timeout: Option<Duration>,
}

impl ProviderAudioTranscriptionRequest {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn body(&self) -> &Value {
        &self.body
    }

    pub fn with_body(self, body: Value) -> Self {
        Self { body, ..self }
    }
}
