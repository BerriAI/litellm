use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::CoreResult;
use litellm_core::audio_transcription::transformation::AudioTranscriptionAuth;
use litellm_core::call_lifecycle::CallContext;
use litellm_core::error::CoreError;
use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{audio_transcription_provider_config, has_header, string_headers};
use super::handler::environment_lookup;
use super::types::{
    AudioTranscriptionRequest, PreparedAudioTranscriptionRequest, ProviderAudioTranscriptionRequest,
};

pub(crate) struct PreparedAudioTranscriptionCall {
    pub(crate) context: CallContext,
    pub(crate) request: PreparedAudioTranscriptionRequest,
}

pub(crate) fn prepare_audio_transcription_call(
    request: AudioTranscriptionRequest<'_>,
) -> PreparedAudioTranscriptionCall {
    let call_id = request
        .litellm_call_id
        .map(str::to_string)
        .unwrap_or_else(new_audio_transcription_call_id);
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "bedrock",
        });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let context = CallContext::new(&model, &custom_llm_provider, call_id);
    let (private_params, optional_params) = request
        .optional_params
        .into_iter()
        .partition(|(key, _)| is_private_param(key));

    PreparedAudioTranscriptionCall {
        context,
        request: PreparedAudioTranscriptionRequest {
            model,
            custom_llm_provider,
            audio: request.audio,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params,
            private_params,
            timeout: request.timeout,
        },
    }
}

pub(crate) async fn prepare_provider_request(
    request: PreparedAudioTranscriptionRequest,
) -> CoreResult<ProviderAudioTranscriptionRequest> {
    if let Some(key) = request
        .optional_params
        .keys()
        .find(|key| is_private_param(key))
    {
        return Err(CoreError::invalid_request(format!(
            "audio transcription interceptor cannot set private parameter '{key}'"
        )));
    }
    let optional_params = request
        .private_params
        .into_iter()
        .chain(request.optional_params)
        .collect();
    let config = audio_transcription_provider_config(&request.custom_llm_provider)
        .ok_or_else(|| CoreError::invalid_provider(request.custom_llm_provider.clone()))?;
    let headers = string_headers(request.extra_headers)?;
    let url = config.complete_url(
        request.api_base.as_deref(),
        &request.model,
        &optional_params,
        &environment_lookup,
    )?;
    let filtered_params = config.map_transcription_params(&optional_params);
    let body =
        config.transform_transcription_request(&request.model, request.audio, filtered_params)?;
    let auth = config.auth_strategy(&request.model, &optional_params, &environment_lookup)?;
    let mut upstream_headers = headers.into_iter().collect::<Vec<_>>();
    if matches!(auth, AudioTranscriptionAuth::Bearer)
        && !has_header(
            &upstream_headers
                .iter()
                .cloned()
                .collect::<std::collections::BTreeMap<_, _>>(),
            "authorization",
        )
        && let Some(api_key) = request.api_key.as_deref()
    {
        upstream_headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
    }

    Ok(ProviderAudioTranscriptionRequest {
        model: request.model,
        custom_llm_provider: request.custom_llm_provider,
        config,
        url,
        body: body.body,
        upstream_headers,
        optional_params,
        timeout: request.timeout,
    })
}

fn is_private_param(key: &str) -> bool {
    key.starts_with("aws_")
}

fn new_audio_transcription_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("audio-transcription-{timestamp}-{sequence}")
}
