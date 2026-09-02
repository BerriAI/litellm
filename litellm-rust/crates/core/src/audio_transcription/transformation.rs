use crate::Error;
use serde_json::{Map, Value};

use super::types::{AudioTranscriptionRequestData, AudioTranscriptionResponseData};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioTranscriptionAuth {
    Bearer,
    AwsSigV4 {
        region: String,
        service: &'static str,
    },
}

pub trait AudioTranscriptionProviderConfig: Sync {
    fn supported_transcription_params(&self) -> &'static [&'static str];

    fn map_transcription_params(&self, params: &Map<String, Value>) -> Map<String, Value> {
        params
            .iter()
            .filter(|(key, _)| {
                self.supported_transcription_params()
                    .contains(&key.as_str())
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

    fn transform_transcription_request(
        &self,
        model: &str,
        audio: Value,
        optional_params: Map<String, Value>,
    ) -> Result<AudioTranscriptionRequestData, Error>;

    fn transform_transcription_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<AudioTranscriptionResponseData, Error>;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn auth_strategy(
        &self,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<AudioTranscriptionAuth, Error>;
}
