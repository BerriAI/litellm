use crate::auth::RequestAuth;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::AudioTranscriptionProviderConfig;

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
    pub(super) config: &'static dyn AudioTranscriptionProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) auth: RequestAuth,
    #[cfg(feature = "bedrock-auth")]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TranscriptionAudio {
    pub data: String,
    pub format: AudioFormat,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TranscriptionParams {
    pub language: Option<String>,
    pub prompt: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::http_utils::deserialize_optional_param"
    )]
    pub temperature: Option<Option<serde_json::Number>>,
    pub response_format: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum AudioTranscriptionRequestData {
    Bedrock(crate::providers::bedrock::converse::ConverseRequest),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ProviderTranscriptionResponse {
    Bedrock(crate::providers::bedrock::converse::ConverseResponse),
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
