use crate::integrations::types::RequestHooks;
use litellm_core::call_lifecycle::CallLifecycleContext;
use litellm_core::request_context::LiteLlmRequestContext;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::hooks::AudioTranscriptionLifecycleHooks;
use super::types::{AudioTranscriptionRequest, PreparedAudioTranscriptionRequest};
use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedAudioTranscriptionCall {
    pub(crate) context: CallLifecycleContext,
    pub(crate) request: PreparedAudioTranscriptionRequest,
    pub(crate) hooks: AudioTranscriptionLifecycleHooks,
}

pub(crate) fn prepare_audio_transcription_call(
    request: AudioTranscriptionRequest<'_>,
    context: &LiteLlmRequestContext,
    hooks: RequestHooks,
) -> PreparedAudioTranscriptionCall {
    let call_id = context
        .litellm_call_id
        .clone()
        .unwrap_or_else(new_audio_transcription_call_id);
    let provider_info = get_custom_llm_provider(
        request.model,
        request.options.custom_llm_provider.as_deref(),
    )
    .unwrap_or(CustomLlmProvider {
        model: request.model,
        custom_llm_provider: "bedrock",
    });
    PreparedAudioTranscriptionCall {
        context: CallLifecycleContext::new(
            "audio_transcription",
            provider_info.model,
            provider_info.custom_llm_provider,
            call_id,
        ),
        request: PreparedAudioTranscriptionRequest {
            model: provider_info.model.to_string(),
            custom_llm_provider: provider_info.custom_llm_provider.to_string(),
            audio: request.audio,
            provider_connection: request.options.provider_connection,
            api_key: request.options.api_key,
            api_base: request.options.api_base,
            extra_headers: request.options.extra_headers,
            optional_params: request.optional_params,
            timeout: request.options.timeout,
        },
        hooks: AudioTranscriptionLifecycleHooks::new(
            CustomLoggerRunner::new(hooks.callbacks),
            CustomGuardrailRunner::new(hooks.guardrails),
            context.attribution.clone(),
        ),
    }
}

fn new_audio_transcription_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("audio-transcription-{timestamp}-{sequence}")
}
