use crate::Error;
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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_transcription_params(&self, params: &OpaqueParams) -> OpaqueParams {
        params.retain_supported(self.supported_transcription_params())
    }

    fn transform_transcription_request(
        &self,
        model: &str,
        audio: Value,
        optional_params: OpaqueParams,
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
        optional_params: &OpaqueParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn auth_strategy(
        &self,
        model: &str,
        optional_params: &OpaqueParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<AudioTranscriptionAuth, Error>;
}
