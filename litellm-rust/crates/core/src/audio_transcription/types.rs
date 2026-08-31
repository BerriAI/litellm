use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::{AudioTranscriptionAuth, AudioTranscriptionProviderConfig};

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

pub(super) struct ProviderAudioTranscriptionRequest {
    pub(super) model: String,
    pub(super) config: &'static dyn AudioTranscriptionProviderConfig,
    pub(super) url: String,
    pub(super) body: Value,
    pub(super) upstream_headers: Vec<(String, String)>,
    pub(super) auth: AudioTranscriptionAuth,
    #[cfg(feature = "bedrock-auth")]
    pub(super) optional_params: Map<String, Value>,
    pub(super) timeout: Option<Duration>,
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
