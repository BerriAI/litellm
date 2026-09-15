use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use litellm_core::Error;
use litellm_core::audio_transcription::lifecycle::{
    AudioTranscriptionCall, OwnedAudioTranscriptionRequest,
};
use litellm_core::call_lifecycle::CallLifecycleRequest;
use litellm_core::call_lifecycle::host::{LifecycleBackend, LifecycleBackendFuture, drive};
use litellm_core::call_lifecycle::provider::{
    CompletedOperation, CompletedReply, CompletedWorkflow, ProviderHooks, ProviderOptions,
};
use litellm_core::call_lifecycle::workflow::LifecycleOperation;

mod hooks;
mod prepare;
mod types;

pub use types::AudioTranscriptionRequest;

use hooks::AudioTranscriptionLifecycleHooks;
use prepare::{PreparedAudioTranscriptionCall, prepare_audio_transcription_call};
use types::PreparedAudioTranscriptionRequest;

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    let PreparedAudioTranscriptionCall { request, hooks } =
        prepare_audio_transcription_call(request);
    let mut call = AudioTranscriptionCall::new(CompletedWorkflow::default(), false);
    drive(
        &mut call,
        &AudioTranscriptionBackend {
            request: Mutex::new(Some(request)),
            hooks,
        },
    )
    .await
}

struct AudioTranscriptionBackend {
    request: Mutex<Option<PreparedAudioTranscriptionRequest>>,
    hooks: AudioTranscriptionLifecycleHooks,
}

impl LifecycleBackend<CompletedOperation<Value>, CompletedReply<OwnedAudioTranscriptionRequest>>
    for AudioTranscriptionBackend
{
    fn invoke(
        &self,
        operation: CompletedOperation<Value>,
    ) -> LifecycleBackendFuture<'_, CompletedReply<OwnedAudioTranscriptionRequest>> {
        Box::pin(async move {
            match operation {
                CompletedOperation::Lifecycle(LifecycleOperation::ProjectRequest) => {
                    let request = self
                        .request
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take()
                        .ok_or_else(|| {
                            Error::InvalidRequest("request was already projected".into())
                        });
                    CompletedReply::Request(match request {
                        Ok(request) => {
                            let context = request.lifecycle_context();
                            let started = epoch_seconds();
                            match self.hooks.run_pre_call_guardrails(request).await {
                                Ok(request) => Ok(owned_request(request)),
                                Err(error) => {
                                    let timing =
                                        litellm_core::call_lifecycle::CallLifecycleTiming::new(
                                            started,
                                            epoch_seconds(),
                                        );
                                    self.hooks.log_failure(&context, &error, &timing).await;
                                    Err(error)
                                }
                            }
                        }
                        Err(error) => Err(error),
                    })
                }
                CompletedOperation::Lifecycle(LifecycleOperation::Success {
                    context,
                    response,
                    timing,
                }) => {
                    self.hooks
                        .log_success(&context, response.as_ref(), &timing)
                        .await;
                    CompletedReply::Lifecycle(Ok(()))
                }
                CompletedOperation::Lifecycle(LifecycleOperation::Failure {
                    context,
                    error,
                    timing,
                }) => {
                    self.hooks.log_failure(&context, &error, &timing).await;
                    CompletedReply::Lifecycle(Ok(()))
                }
                CompletedOperation::Lifecycle(_) => CompletedReply::Lifecycle(Ok(())),
                CompletedOperation::BeforeRequest(request) => {
                    CompletedReply::BeforeRequest(self.hooks.before_request(request).await)
                }
                CompletedOperation::AfterResponse(response) => {
                    CompletedReply::AfterResponse(self.hooks.after_response(response).await)
                }
            }
        })
    }
}

fn owned_request(request: PreparedAudioTranscriptionRequest) -> OwnedAudioTranscriptionRequest {
    OwnedAudioTranscriptionRequest {
        options: ProviderOptions {
            model: request.model,
            litellm_call_id: Some(request.litellm_call_id),
            api_key: request.api_key,
            api_base: request.api_base,
            custom_llm_provider: Some(request.custom_llm_provider),
            extra_headers: request.extra_headers,
            timeout: request.timeout,
        },
        audio: request.audio,
        optional_params: request.optional_params,
    }
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests;
