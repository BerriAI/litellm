use litellm_core::CoreResult;
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod client;
mod common_utils;
mod handler;
mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use handler::execute_audio_transcription_provider_call;
use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> CoreResult<Value> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, execute_audio_transcription_provider_call)
        .await
}

#[cfg(test)]
mod tests;
