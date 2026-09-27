use std::time::Duration;

use litellm_http::{Client, request::truncate_error_body};
use litellm_llms::base_llm::auth::resolve_auth;
use serde_json::Value;

use super::Error;
use crate::{
    audio_transcription::types::ProviderAudioTranscriptionRequest,
    constants::AUDIO_TRANSCRIPTION_TIMEOUT_SECS,
};

pub async fn execute_audio_transcription_provider_call(
    http: &Client,
    auth: &litellm_auth::AuthServices,
    request: ProviderAudioTranscriptionRequest,
) -> Result<Value, Error> {
    let env_lookup = |key: &str| std::env::var(key).ok();
    let authenticated = resolve_auth(auth, request.environment.clone(), &env_lookup).await?;
    let response = crate::outbound::outbound_request(
        authenticated,
        request.url.clone(),
        &request.body,
        Some(
            request
                .timeout
                .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
        ),
    )?
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
