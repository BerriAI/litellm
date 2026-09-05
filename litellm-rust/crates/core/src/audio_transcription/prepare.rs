use crate::error::Error;
use crate::http_utils::{has_header, string_headers};
#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{AudioTranscriptionAuth, AudioTranscriptionProviderConfig};
use super::types::{AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn provider_config(provider: &str) -> Option<&'static dyn AudioTranscriptionProviderConfig> {
    #[cfg(feature = "bedrock-auth")]
    if provider == "bedrock" {
        return Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG);
    }
    let _ = provider;
    None
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub fn prepare_audio_transcription_provider_call(
    request: AudioTranscriptionRequest<'_>,
) -> Result<ProviderAudioTranscriptionRequest, Error> {
    let provider_info = get_custom_llm_provider(
        request.model,
        request.options.custom_llm_provider.as_deref(),
    )
    .or_else(|| {
        request
            .options
            .custom_llm_provider
            .as_deref()
            .map(|provider| CustomLlmProvider {
                model: request.model,
                custom_llm_provider: provider,
            })
    })
    .ok_or_else(|| {
        Error::InvalidProvider(
            "unable to resolve custom_llm_provider for audio transcription request".to_string(),
        )
    })?;
    let model = provider_info.model.to_string();
    let config = provider_config(provider_info.custom_llm_provider)
        .ok_or_else(|| Error::InvalidProvider(provider_info.custom_llm_provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let mut headers = string_headers("audio transcription", request.options.extra_headers)?;
    let auth = config.auth_strategy(&model, &request.options.provider_connection, &env_lookup)?;
    if matches!(auth, AudioTranscriptionAuth::Bearer)
        && !has_header(&headers, "authorization")
        && let Some(api_key) = request.options.api_key.as_deref()
    {
        headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
    }
    if !has_header(&headers, "content-type") {
        headers.push(("Content-Type".to_string(), "application/json".to_string()));
    }
    let url = config.complete_url(
        request.options.api_base.as_deref(),
        &model,
        &request.options.provider_connection,
        &env_lookup,
    )?;
    let filtered_params = config.map_transcription_params(&request.optional_params);
    let transformed =
        config.transform_transcription_request(&model, request.audio, filtered_params)?;
    Ok(ProviderAudioTranscriptionRequest {
        model,
        custom_llm_provider: provider_info.custom_llm_provider.to_string(),
        config,
        url,
        body: transformed.body,
        upstream_headers: headers,
        auth,
        #[cfg(feature = "bedrock-auth")]
        provider_connection: request.options.provider_connection,
        timeout: request.options.timeout,
    })
}
