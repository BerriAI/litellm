use litellm_core::Error;
use litellm_core::audio_transcription::execute_audio_transcription_provider_call;
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, execute_audio_transcription_provider_call)
        .await
}

#[cfg(test)]
mod tests;
