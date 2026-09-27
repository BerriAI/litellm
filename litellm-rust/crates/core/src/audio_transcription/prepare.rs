use litellm_core_utils::get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider};
use litellm_http::request::string_headers;
use litellm_llms::{
    base_llm::{
        audio_transcription::transformation::BaseAudioTranscriptionConfig,
        auth::{ValidatedEnvironment, with_default_headers},
    },
    bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG,
};

use super::Error;
use crate::audio_transcription::types::{
    AudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
};

fn provider_config(provider: &str) -> Option<&'static dyn BaseAudioTranscriptionConfig> {
    if provider == "bedrock" {
        return Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG);
    }
    let _ = provider;
    None
}

pub fn prepare_audio_transcription_provider_call(
    request: AudioTranscriptionRequest<'_>,
) -> Result<ProviderAudioTranscriptionRequest, Error> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
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
    let forwarded = string_headers("audio transcription", request.extra_headers)?;
    let validated =
        config.validate_environment(forwarded, &model, &request.optional_params, &env_lookup)?;
    let environment = ValidatedEnvironment {
        headers: with_default_headers(validated.headers, &[("Content-Type", "application/json")]),
        auth: validated.auth,
    };
    let url = config.get_complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    let filtered_params = config.map_transcription_params(&request.optional_params);
    let transformed =
        config.transform_audio_transcription_request(&model, request.audio, filtered_params)?;
    Ok(ProviderAudioTranscriptionRequest {
        model,
        custom_llm_provider: provider_info.custom_llm_provider.to_string(),
        config,
        url,
        body: transformed.body,
        environment,
        timeout: request.timeout,
    })
}
