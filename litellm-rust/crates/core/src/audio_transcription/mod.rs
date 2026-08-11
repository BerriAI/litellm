//! Audio transcription SDK entrypoint.
//!
//! Provider resolution, request transformation, authentication, transport, and
//! response transformation live in core so every host uses the same execution
//! path.

mod client;
mod common_utils;
mod handler;
mod prepare;

pub mod transformation;
pub mod types;

use crate::CoreResult;

use handler::execute_audio_transcription_provider_call;
use prepare::prepare_audio_transcription_call;

pub use types::{
    AudioTranscriptionRequest, AudioTranscriptionResponseData, PreparedAudioTranscriptionRequest,
};

/// Transform a host request into a provider-specific request.
///
/// The returned value deliberately exposes only the information required by
/// host lifecycle integrations. Authentication is finalized during execution
/// so body mutations made by guardrails are included in provider signatures.
pub fn prepare_audio_transcription_request(
    request: AudioTranscriptionRequest<'_>,
) -> CoreResult<PreparedAudioTranscriptionRequest> {
    prepare_audio_transcription_call(request)
}

/// Execute a previously prepared audio transcription request.
///
/// Authentication and signing happen here, immediately before transport.
pub async fn execute_prepared_audio_transcription(
    request: PreparedAudioTranscriptionRequest,
) -> CoreResult<AudioTranscriptionResponseData> {
    execute_audio_transcription_provider_call(request).await
}

/// Execute an audio transcription request through the resolved provider.
pub async fn audio_transcription(
    request: AudioTranscriptionRequest<'_>,
) -> CoreResult<AudioTranscriptionResponseData> {
    execute_prepared_audio_transcription(prepare_audio_transcription_request(request)?).await
}

#[cfg(all(test, feature = "bedrock-auth"))]
mod tests;
