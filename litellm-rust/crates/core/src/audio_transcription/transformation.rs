use serde_json::Value;

use super::types::{AudioTranscriptionRequestData, AudioTranscriptionResponseData};
use crate::params::OpaqueParams;

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

    fn map_transcription_params(&self, params: &OpaqueParams) -> OpaqueParams {
        params.provider_params()
    }

    fn transform_transcription_request(
        &self,
        model: &str,
        audio: Value,
        optional_params: OpaqueParams,
    ) -> Result<AudioTranscriptionRequestData, super::Error>;

    fn transform_transcription_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<AudioTranscriptionResponseData, super::Error>;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &OpaqueParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, super::Error>;

    fn auth_strategy(
        &self,
        model: &str,
        optional_params: &OpaqueParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<AudioTranscriptionAuth, super::Error>;
}
