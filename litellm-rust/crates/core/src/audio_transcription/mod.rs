mod error;
pub use error::Error;
mod client;
mod handler;
mod prepare;
pub use litellm_providers::audio_transcription::types;

pub use handler::execute_audio_transcription_provider_call;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;
pub use types::{AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    execute_audio_transcription_provider_call(prepare_audio_transcription_provider_call(request)?)
        .await
}

#[cfg(test)]
mod tests;
