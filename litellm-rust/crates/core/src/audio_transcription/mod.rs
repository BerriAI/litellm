use crate::Error;
mod client;
mod handler;
pub mod lifecycle;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

pub use handler::execute_audio_transcription_provider_call;
pub use prepare::prepare_audio_transcription_provider_call;
pub use types::{AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

pub struct AudioTranscriptionAdmission {
    pub model: String,
    pub provider: Option<String>,
    pub audio: Value,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    crate::call_lifecycle::provider::run_completed::<lifecycle::AudioTranscriptionRoute>(
        request.into(),
    )
    .await
}

pub fn admit(
    inspection: crate::call_lifecycle::admission::Inspection<AudioTranscriptionAdmission>,
) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
    use crate::call_lifecycle::admission::{AdmissionDecline, Inspection};
    let Inspection::Inspectable(admission) = inspection else {
        return Err(AdmissionDecline::Uninspectable);
    };
    let resolved = crate::routing_utils::provider::get_custom_llm_provider(
        &admission.model,
        admission.provider.as_deref(),
    );
    let provider = admission
        .provider
        .as_deref()
        .or_else(|| resolved.as_ref().map(|value| value.custom_llm_provider));
    if provider.and_then(prepare::provider_config).is_none() {
        return Err(AdmissionDecline::Provider);
    }
    if let Some(format) = admission.audio.get("format").and_then(Value::as_str)
        && !matches!(format, "wav" | "mp3" | "flac" | "ogg")
    {
        return Err(AdmissionDecline::Feature("unsupported audio format"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
