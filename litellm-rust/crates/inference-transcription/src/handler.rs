use std::time::Duration;

use litellm_http::Client;
use litellm_inference::call::{self, Failure};
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
) -> Result<Value, Failure<Error>> {
    let outbound = call::prepare(async {
        let env_lookup = |key: &str| request.secrets.get(key);
        let authenticated = resolve_auth(auth, request.environment.clone(), &env_lookup).await?;
        litellm_inference::outbound::outbound_request(
            authenticated,
            request.url.clone(),
            &request.body,
            Some(
                request
                    .timeout
                    .unwrap_or(Duration::from_secs(AUDIO_TRANSCRIPTION_TIMEOUT_SECS)),
            ),
        )
        .map_err(Error::from)
    })
    .await?;
    let response = call::send(http, outbound).await?;
    call::receive(async {
        let text = response
            .text()
            .await
            .map_err(|error| Error::Transport(error.into()))?;
        let response_json = serde_json::from_str(&text).map_err(|error| {
            Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
                "audio response JSON",
                error,
            ))
        })?;
        Ok(request
            .config
            .transform_audio_transcription_response(&request.model, response_json)?
            .into_json())
    })
    .await
}
