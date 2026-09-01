mod client;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::error::Error;

use handler::execute_audio_transcription_provider_call;
use prepare::prepare_audio_transcription_call;
pub use types::AudioTranscriptionRequest;

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    execute_audio_transcription_provider_call(prepare_audio_transcription_call(request)?).await
}

#[cfg(test)]
mod tests;
