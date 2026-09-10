use crate::auth::RequestAuth;
use crate::error::Error;
use crate::http_utils::{has_header, string_headers};
#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::AudioTranscriptionProviderConfig;
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
    let mut headers = string_headers("audio transcription", request.extra_headers)?;
    let auth = config.auth(
        request.api_key,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    match &auth {
        RequestAuth::Bearer { token } if !has_header(&headers, "authorization") => {
            headers.push(("Authorization".into(), format!("Bearer {token}")));
        }
        RequestAuth::Header { name, value } if !has_header(&headers, name) => {
            headers.push(((*name).into(), value.clone()));
        }
        _ => {}
    }
    if !has_header(&headers, "content-type") {
        headers.push(("Content-Type".to_string(), "application/json".to_string()));
    }
    let url = config.complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    let audio = serde_json::from_value(request.audio)
        .map_err(|_| Error::InvalidRequest("invalid audio data or format".into()))?;
    let params = map_params(request.optional_params.clone())?;
    let transformed = config.transform_request(&model, audio, params)?;
    let body = serde_json::to_value(transformed)
        .map_err(|_| Error::InvalidRequest("invalid audio request body".into()))?;
    Ok(ProviderAudioTranscriptionRequest {
        model,
        custom_llm_provider: provider_info.custom_llm_provider.to_string(),
        config,
        url,
        body,
        upstream_headers: headers,
        auth,
        #[cfg(feature = "bedrock-auth")]
        optional_params: request.optional_params,
        timeout: request.timeout,
    })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn map_params(
    params: serde_json::Map<String, serde_json::Value>,
) -> Result<super::types::TranscriptionParams, Error> {
    serde_json::from_value(serde_json::Value::Object(params))
        .map_err(|_| Error::InvalidRequest("invalid transcription parameters".into()))
}
