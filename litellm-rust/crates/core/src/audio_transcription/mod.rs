use crate::Error;
mod client;
mod handler;
mod lifecycle;
mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;

pub use lifecycle::{AudioRoute, AudioServices, DefaultAudioServices};
pub use types::{AudioRouteRequest, AudioTranscriptionRequest, ProviderAudioTranscriptionRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    let services = DefaultAudioServices::new(Vec::new(), Vec::new());
    AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model: request.model,
            audio: request.audio,
            api_key: request.api_key,
            api_base: request.api_base,
            custom_llm_provider: request.custom_llm_provider,
            extra_headers: request.extra_headers,
            optional_params: request.optional_params,
            timeout: request.timeout,
            request_metadata: Default::default(),
            litellm_call_id: None,
        },
    )
    .await
    .into_result()
}

#[cfg(test)]
mod tests;
