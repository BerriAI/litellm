use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;

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

pub use crate::base_llm::auth::{Headers, ValidatedEnvironment};

pub trait BaseAudioTranscriptionConfig: Sync {
    fn get_supported_openai_params(&self) -> &'static [&'static str];

    fn map_transcription_params(
        &self,
        non_default_params: &Map<String, Value>,
    ) -> Map<String, Value> {
        non_default_params
            .iter()
            .filter(|(key, _)| self.get_supported_openai_params().contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_audio_transcription_request(
        &self,
        model: &str,
        audio: Value,
        optional_params: Map<String, Value>,
    ) -> Result<AudioTranscriptionRequestData, Error>;

    fn transform_audio_transcription_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<AudioTranscriptionResponseData, Error>;

    fn validate_environment(
        &self,
        headers: Headers,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error>;
}
