use std::time::Duration;

use litellm_http::{Client, request::truncate_error_body};
use serde_json::Value;

use super::Error;
use crate::{
    audio_transcription::types::ProviderAudioTranscriptionRequest,
    constants::AUDIO_TRANSCRIPTION_TIMEOUT_SECS,
};

pub async fn execute_audio_transcription_provider_call(
    http: &Client,
    request: ProviderAudioTranscriptionRequest,
) -> Result<Value, Error> {
    let response = crate::outbound::outbound_request::<Error>(
        &request.auth,
        request.url.clone(),
        request.upstream_headers.clone(),
        &request.body,
        Some(
            request
                .timeout
                .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
        ),
        &request.optional_params,
    )
    .await?
    .send(http)
    .await
    .map_err(|error| {
        Error::Transport(litellm_http::transport::Error::Network(error.to_string()))
    })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        Error::Transport(litellm_http::transport::Error::Network(error.to_string()))
    })?;
    if !status.is_success() {
        return Err(Error::Transport(litellm_http::transport::Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        }));
    }
    let response_json = serde_json::from_str(&text)
        .map_err(|error| Error::InvalidResponse(format!("invalid audio response JSON: {error}")))?;
    Ok(request
        .config
        .transform_audio_transcription_response(&request.model, response_json)?
        .into_json())
}
