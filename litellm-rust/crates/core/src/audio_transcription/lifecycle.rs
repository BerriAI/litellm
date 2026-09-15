use std::sync::Arc;

use serde_json::Value;

use super::types::AudioTranscriptionRequest;
use crate::call_lifecycle::provider::{
    CompletedCall, CompletedRoute, ProviderHooks, ProviderOptions,
};
use crate::call_lifecycle::workflow::WorkflowFuture;

pub struct OwnedAudioTranscriptionRequest {
    pub options: ProviderOptions,
    pub audio: Value,
    pub optional_params: serde_json::Map<String, Value>,
}

impl From<AudioTranscriptionRequest<'_>> for OwnedAudioTranscriptionRequest {
    fn from(request: AudioTranscriptionRequest<'_>) -> Self {
        Self {
            options: ProviderOptions {
                model: request.model.to_owned(),
                litellm_call_id: None,
                api_key: request.api_key.map(str::to_owned),
                api_base: request.api_base.map(str::to_owned),
                custom_llm_provider: request.custom_llm_provider.map(str::to_owned),
                extra_headers: request.extra_headers,
                timeout: request.timeout,
            },
            audio: request.audio,
            optional_params: request.optional_params,
        }
    }
}

pub struct AudioTranscriptionRoute;
pub type AudioTranscriptionCall = CompletedCall<AudioTranscriptionRoute>;

impl CompletedRoute for AudioTranscriptionRoute {
    type Admission =
        crate::call_lifecycle::admission::Inspection<super::AudioTranscriptionAdmission>;
    type Request = OwnedAudioTranscriptionRequest;
    type Response = Value;

    fn admit(
        admission: Self::Admission,
    ) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
        super::admit(admission)
    }

    fn run(
        request: Self::Request,
        hooks: Arc<dyn ProviderHooks>,
    ) -> WorkflowFuture<Self::Response> {
        Box::pin(async move {
            let options = request.options;
            let request = AudioTranscriptionRequest {
                model: &options.model,
                api_key: options.api_key.as_deref(),
                api_base: options.api_base.as_deref(),
                custom_llm_provider: options.custom_llm_provider.as_deref(),
                extra_headers: options.extra_headers,
                timeout: options.timeout,
                audio: request.audio,
                optional_params: request.optional_params,
            };
            super::handler::execute_with_hooks(
                super::prepare::prepare_audio_transcription_provider_call(request)?,
                hooks.as_ref(),
            )
            .await
        })
    }

    fn context(request: &Self::Request) -> crate::call_lifecycle::CallLifecycleContext {
        request.options.lifecycle_context("audio_transcription")
    }
}
