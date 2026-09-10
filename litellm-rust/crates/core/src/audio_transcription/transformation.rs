use crate::Error;
use crate::auth::RequestAuth;
use serde_json::{Map, Value};

use super::types::{
    AudioTranscriptionRequestData, AudioTranscriptionResponseData, ProviderTranscriptionResponse,
    TranscriptionAudio, TranscriptionParams,
};

pub trait AudioTranscriptionProviderConfig: Sync {
    fn auth(
        &self,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<RequestAuth, Error>;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_request(
        &self,
        model: &str,
        audio: TranscriptionAudio,
        params: TranscriptionParams,
    ) -> Result<AudioTranscriptionRequestData, Error>;

    fn transform_response(
        &self,
        model: &str,
        response: ProviderTranscriptionResponse,
    ) -> Result<AudioTranscriptionResponseData, Error>;
}
