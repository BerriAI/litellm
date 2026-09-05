use crate::error::Error;
use crate::http_utils::{has_header, string_headers};
#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use crate::request_options::RequestOptions;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{AudioTranscriptionAuth, AudioTranscriptionProviderConfig};
use super::types::{AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) fn provider_config(
    provider: &str,
) -> Option<&'static dyn AudioTranscriptionProviderConfig> {
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
    options: RequestOptions,
) -> Result<ProviderAudioTranscriptionRequest, Error> {
    let provider_info =
        get_custom_llm_provider(request.model, options.custom_llm_provider.as_deref())
            .or_else(|| {
                options
                    .custom_llm_provider
                    .as_deref()
                    .map(|provider| CustomLlmProvider {
                        model: request.model,
                        custom_llm_provider: provider,
                    })
            })
            .ok_or_else(|| {
                Error::InvalidProvider(
                    "unable to resolve custom_llm_provider for audio transcription request"
                        .to_string(),
                )
            })?;
    let model = provider_info.model.to_string();
    let config = provider_config(provider_info.custom_llm_provider)
        .ok_or_else(|| Error::InvalidProvider(provider_info.custom_llm_provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let mut headers = string_headers("audio transcription", options.extra_headers)?;
    let bedrock = options.bedrock.unwrap_or_default();
    let bedrock_options = bedrock.clone().into_map();
    let auth = config.auth_strategy(&model, &bedrock_options, &env_lookup)?;
    if matches!(auth, AudioTranscriptionAuth::Bearer)
        && !has_header(&headers, "authorization")
        && let Some(api_key) = options.api_key.as_deref()
    {
        headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
    }
    if !has_header(&headers, "content-type") {
        headers.push(("Content-Type".to_string(), "application/json".to_string()));
    }
    let url = config.complete_url(
        options.api_base.as_deref(),
        &model,
        &bedrock_options,
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
        bedrock,
        timeout: options.timeout,
    })
}
