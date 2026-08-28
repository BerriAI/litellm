//! The audio transcription call, the Rust equivalent of Python's
//! `litellm.transcription()`.
//!
//! [`audio_transcription`] is the top-level entrypoint: give it a model, an
//! audio payload, and credentials, and it resolves the provider, transforms
//! the request, calls the provider, and returns the normalized response JSON.

mod client;
mod common_utils;
mod handler;
mod hooks;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::CoreResult;
use crate::call_lifecycle::CallLifecycle;

use handler::execute_audio_transcription_provider_call;
use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};
pub use types::AudioTranscriptionRequest;

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> CoreResult<Value> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, execute_audio_transcription_provider_call)
        .await
}

#[cfg(test)]
mod tests;
