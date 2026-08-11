use litellm_core::CoreResult;
use litellm_core::audio_transcription::{
    PreparedAudioTranscriptionRequest as CorePreparedAudioTranscriptionRequest,
    execute_prepared_audio_transcription,
};
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};

async fn execute_audio_transcription(
    request: CorePreparedAudioTranscriptionRequest,
) -> CoreResult<Value> {
    execute_prepared_audio_transcription(request)
        .await
        .map(|response| response.into_json())
}

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> CoreResult<Value> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);

    CallLifecycle::default()
        .run_request(request, &hooks, execute_audio_transcription)
        .await
}

#[cfg(test)]
mod tests;
