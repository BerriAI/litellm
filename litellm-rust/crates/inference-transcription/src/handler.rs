use std::time::Duration;

use litellm_http::{
    Client,
    request::{string_headers, truncate_error_body, with_default_headers},
};
use litellm_llms::base_llm::{
    audio_transcription::transformation::BaseAudioTranscriptionConfig,
    auth::{ValidatedEnvironment, resolve_auth},
};
use litellm_secrets::source::Secrets;
use serde_json::Value;

use super::Error;
use crate::{constants::AUDIO_TRANSCRIPTION_TIMEOUT_SECS, types::AudioTranscriptionRequest};
use litellm_inference::provider::ResolvedProvider;

pub(super) async fn execute(
    http: &Client,
    auth: &litellm_auth::AuthServices,
    config: &'static dyn BaseAudioTranscriptionConfig,
    provider: ResolvedProvider<'_>,
    call: AudioTranscriptionRequest<'_>,
    secrets: Secrets,
) -> Result<Value, Error> {
    let env_lookup = |key: &str| secrets.get(key);
    let forwarded = string_headers("audio transcription", call.extra_headers)?;
    let validated = config.validate_environment(
        forwarded,
        provider.model,
        &call.optional_params,
        &env_lookup,
    )?;
    let environment = ValidatedEnvironment {
        headers: with_default_headers(validated.headers, config.default_headers()),
        auth: validated.auth,
    };
    let url = config.get_complete_url(
        call.api_base,
        provider.model,
        &call.optional_params,
        &env_lookup,
    )?;
    let filtered_params = config.map_transcription_params(&call.optional_params);
    let body = config
        .transform_audio_transcription_request(provider.model, call.audio, filtered_params)?
        .body;
    let authenticated = resolve_auth(auth, environment, &env_lookup).await?;
    let outbound = litellm_inference::outbound::outbound_request(
        authenticated,
        url,
        &body,
        Some(
            call.timeout
                .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
        ),
    )?;
    let response = litellm_inference::outbound::send(outbound, http)
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
    let response_json = serde_json::from_str(&text).map_err(|error| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "audio response JSON",
            error,
        ))
    })?;
    Ok(config
        .transform_audio_transcription_response(provider.model, response_json)?
        .into_json())
}
