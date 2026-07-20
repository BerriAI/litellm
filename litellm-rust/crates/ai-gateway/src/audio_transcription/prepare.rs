use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::hooks::AudioTranscriptionLifecycleHooks;
use super::types::{AudioTranscriptionRequest, PreparedAudioTranscriptionRequest};
use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;

pub(crate) struct PreparedAudioTranscriptionCall {
    pub(crate) request: PreparedAudioTranscriptionRequest,
    pub(crate) hooks: AudioTranscriptionLifecycleHooks,
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
    PreparedAudioTranscriptionCall {
        request: PreparedAudioTranscriptionRequest {
            model: provider_info.model.to_string(),
            custom_llm_provider: provider_info.custom_llm_provider.to_string(),
            litellm_call_id: call_id,
            audio: request.audio,
            api_key: request.api_key.map(str::to_string),
            api_base: request.api_base.map(str::to_string),
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: request.timeout,
        },
        hooks: AudioTranscriptionLifecycleHooks::new(
            CustomLoggerRunner::new(request.callbacks),
            CustomGuardrailRunner::new(request.guardrails),
            request.request_metadata,
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
