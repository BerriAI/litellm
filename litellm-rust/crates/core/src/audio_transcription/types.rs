use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::base_llm::audio_transcription::transformation::{
    AudioTranscriptionAuth, BaseAudioTranscriptionConfig,
};

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
    pub(super) model: String,
    pub(super) custom_llm_provider: String,
    pub(super) config: &'static dyn BaseAudioTranscriptionConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) auth: AudioTranscriptionAuth,
    pub(super) optional_params: Map<String, Value>,
    pub(super) timeout: Option<Duration>,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioTranscriptionRequestData {
    pub body: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioTranscriptionResponseData {
    pub text: String,
}

impl AudioTranscriptionResponseData {
    pub fn into_json(self) -> Value {
        serde_json::json!({
            "text": self.text,
        })
    }
}
