use serde_json::{Map, Value};

use crate::audio_transcription::Error;
use crate::audio_transcription::types::{
    AudioTranscriptionRequestData, AudioTranscriptionResponseData,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioTranscriptionAuth {
    Bearer,
    AwsSigV4 {
        region: String,
        service: &'static str,
    },
}

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

    fn auth_strategy(
        &self,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<AudioTranscriptionAuth, Error>;
}
