//! Audio request preparation logic.

use serde_json::Value;

use crate::error::{CoreError, CoreResult};

use super::transformation::get_audio_provider_config;
use super::types::{SpeechRequest, TranscriptionRequest};

/// Prepared speech request ready to send to the provider.
pub struct PreparedSpeechRequest {
    pub url: String,
    pub body: Value,
    pub api_key: Option<String>,
    pub content_type: String,
}

/// Prepared transcription request ready to send to the provider.
pub struct PreparedTranscriptionRequest {
    pub url: String,
    pub body: Value,
    pub file: Vec<u8>,
    pub api_key: Option<String>,
}

/// Prepare a speech request for the given provider.
pub fn prepare_speech_request(
    request: &SpeechRequest,
    provider: &str,
    api_base: Option<&str>,
    api_key: Option<String>,
) -> CoreResult<PreparedSpeechRequest> {
    let config = get_audio_provider_config(provider)?;
    let url = config.speech_url(api_base);
    let body = config.transform_speech_request(request)?;
    let content_type = config.speech_response_content_type(request.response_format.as_deref());

    Ok(PreparedSpeechRequest {
        url,
        body,
        api_key,
        content_type,
    })
}

/// Prepare a transcription request for the given provider.
pub fn prepare_transcription_request(
    request: &TranscriptionRequest,
    provider: &str,
    api_base: Option<&str>,
    api_key: Option<String>,
) -> CoreResult<PreparedTranscriptionRequest> {
    let config = get_audio_provider_config(provider)?;
    let url = config.transcription_url(api_base);
    let body = config.transform_transcription_request(request)?;

    Ok(PreparedTranscriptionRequest {
        url,
        body,
        file: request.file.clone(),
        api_key,
    })
}
