use std::time::Duration;

use litellm_http::Client;
use litellm_llms::base_llm::auth::resolve_auth;
use serde_json::Value;

use super::Error;
use crate::{
    constants::AUDIO_TRANSCRIPTION_TIMEOUT_SECS, types::ProviderAudioTranscriptionRequest,
};

pub async fn execute_audio_transcription_provider_call(
    http: &Client,
    auth: &litellm_auth::AuthServices,
    request: ProviderAudioTranscriptionRequest,
) -> Result<http::Response<Value>, Error> {
    let env_lookup = |key: &str| request.secrets.get(key);
    let authenticated = resolve_auth(auth, request.environment.clone(), &env_lookup).await?;
    let outbound = litellm_inference::outbound::outbound_request(
        authenticated,
        request.url.clone(),
        &request.body,
        Some(
            request
                .timeout
                .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
        ),
    )?;
    let response = litellm_inference::outbound::send(outbound, http).await?;
    let (parts, body) = litellm_inference::outbound::read(response)
        .await?
        .into_parts();
    let response_json = serde_json::from_slice(&body).map_err(|error| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "audio response JSON",
            error,
        ))
    })?;
    Ok(http::Response::from_parts(
        parts,
        request
            .config
            .transform_audio_transcription_response(&request.model, response_json)?
            .into_json(),
    ))
}
