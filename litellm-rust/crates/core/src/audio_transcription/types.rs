use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleRequest};
use crate::callbacks::custom_guardrail::CustomGuardrail;
use crate::callbacks::custom_logger::CustomLogger;
use crate::callbacks::types::RequestMetadata;

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

pub(crate) struct PreparedAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: String,
    pub(crate) litellm_call_id: String,
    pub(crate) audio: Value,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) optional_params: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
}

impl CallLifecycleRequest for PreparedAudioTranscriptionRequest {
    fn lifecycle_context(&self) -> CallLifecycleContext {
        CallLifecycleContext::new(
            "audio_transcription",
            self.model.clone(),
            self.custom_llm_provider.clone(),
            self.litellm_call_id.clone(),
        )
    }
}

#[derive(Clone)]
pub(crate) struct ProviderAudioTranscriptionRequest {
    pub(crate) model: String,
    pub(crate) config: &'static dyn AudioTranscriptionProviderConfig,
    pub(crate) url: String,
    pub(crate) body: Value,
    pub(crate) upstream_headers: Vec<(String, String)>,
    pub(crate) timeout: Option<Duration>,
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
