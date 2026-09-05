use crate::integrations::types::RequestHooks;
use litellm_core::Error;
use litellm_core::audio_transcription::execute_audio_transcription_provider_call;
use litellm_core::call_lifecycle::CallLifecycle;
use litellm_core::request_context::LiteLlmRequestContext;
use litellm_core::request_options::RequestOptions;
use serde_json::Value;

mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};

pub async fn audio_transcription(
    request: AudioTranscriptionRequest<'_>,
    options: &RequestOptions,
    context: &LiteLlmRequestContext,
    hooks: RequestHooks,
) -> Result<Value, Error> {
    let PreparedAudioTranscriptionCall {
        request,
        context,
        hooks,
    } = prepare_audio_transcription_call(request, options.clone(), context, hooks);
    CallLifecycle::default()
        .run(
            context,
            request,
            &hooks,
            execute_audio_transcription_provider_call,
        )
        .await
}

#[cfg(test)]
mod tests;
