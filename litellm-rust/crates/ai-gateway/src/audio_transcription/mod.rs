use litellm_core::Error;
use litellm_core::audio_transcription::{AudioRoute, AudioRouteRequest, DefaultAudioServices};
use serde_json::Value;

mod types;

pub use types::AudioTranscriptionRequest;

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    let AudioTranscriptionRequest {
        model,
        audio,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout,
        callbacks,
        guardrails,
        request_metadata,
        litellm_call_id,
    } = request;
    let services = DefaultAudioServices::new(callbacks, guardrails);
    AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model,
            audio,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            optional_params,
            timeout,
            request_metadata,
            litellm_call_id,
        },
    )
    .await
    .into_result()
}
