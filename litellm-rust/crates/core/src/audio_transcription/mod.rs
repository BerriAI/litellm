use crate::Error;
mod client;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

pub use handler::execute_audio_transcription_provider_call;
pub use prepare::prepare_audio_transcription_provider_call;
pub use types::{AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    execute_audio_transcription_provider_call(prepare_audio_transcription_provider_call(request)?)
        .await
}

#[cfg(test)]
mod tests;
