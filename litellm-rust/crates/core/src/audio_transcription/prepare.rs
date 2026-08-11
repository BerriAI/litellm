use crate::error::{CoreError, CoreResult};
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{audio_transcription_provider_config, has_header, string_headers};
use super::transformation::AudioTranscriptionAuth;
use super::types::{AudioTranscriptionRequest, PreparedAudioTranscriptionRequest};

pub(super) fn prepare_audio_transcription_call(
    request: AudioTranscriptionRequest<'_>,
) -> CoreResult<PreparedAudioTranscriptionRequest> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "bedrock",
        });

    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();

    let config = audio_transcription_provider_config(&custom_llm_provider)
        .ok_or_else(|| CoreError::InvalidProvider(custom_llm_provider.clone()))?;

    let env_lookup = environment_lookup;

    let headers = string_headers(request.extra_headers)?;

    let url = config.complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;

    let filtered_params = config.map_transcription_params(&request.optional_params);

    let transformed =
        config.transform_transcription_request(&model, request.audio, filtered_params)?;

    let auth = config.auth_strategy(&model, &request.optional_params, &env_lookup)?;

    let headers = if matches!(auth, AudioTranscriptionAuth::Bearer)
        && !has_header(&headers, "authorization")
        && let Some(api_key) = request.api_key
    {
        headers
            .into_iter()
            .chain(std::iter::once((
                "Authorization".to_string(),
                format!("Bearer {api_key}"),
            )))
            .collect()
    } else {
        headers
    };

    Ok(PreparedAudioTranscriptionRequest {
        model,
        custom_llm_provider,
        config,
        url,
        body: transformed.body,
        upstream_headers: headers.into_iter().collect(),
        optional_params: request.optional_params,
        timeout: request.timeout,
    })
}

pub(super) fn environment_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}
