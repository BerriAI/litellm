use litellm_core::Error;
use litellm_core::audio_transcription::execute_audio_transcription_provider_call;
use litellm_core::lifecycle::{CallLifecycle, CallLifecycleRequest, SystemClock};
use serde_json::Value;

mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);
    let context = request.lifecycle_context();
    CallLifecycle
        .run(
            context,
            request,
            &hooks,
            &hooks,
            &SystemClock,
            execute_audio_transcription_provider_call,
        )
        .await
        .into_result()
}

#[cfg(test)]
mod tests;
