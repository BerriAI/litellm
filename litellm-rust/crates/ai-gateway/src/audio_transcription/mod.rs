use litellm_core::CoreResult;
use litellm_core::call_lifecycle::{CallInterceptor, CallRuntime, NoopCallInterceptor};
use litellm_core::callbacks::CallbackOptions;
use serde_json::Value;

mod common_utils;
mod handler;
mod hooks;
mod prepare;
mod types;

pub use types::{
    AudioTranscriptionCall, AudioTranscriptionRequest, PreparedAudioTranscriptionRequest,
    ProviderAudioTranscriptionRequest,
};

use handler::execute_audio_transcription_provider_call;
use hooks::AudioTranscriptionCallbackInterceptor;
use prepare::{
    PreparedAudioTranscriptionCall, prepare_audio_transcription_call, prepare_provider_request,
};

pub async fn audio_transcription(request: AudioTranscriptionRequest<'_>) -> CoreResult<Value> {
    audio_transcription_with_interceptor(request, &NoopCallInterceptor).await
}

pub async fn audio_transcription_with_callbacks(
    request: AudioTranscriptionRequest<'_>,
    callbacks: CallbackOptions,
) -> CoreResult<Value> {
    audio_transcription_with_interceptor(
        request,
        &AudioTranscriptionCallbackInterceptor::new(callbacks),
    )
    .await
}

pub async fn audio_transcription_with_interceptor<Interceptor>(
    request: AudioTranscriptionRequest<'_>,
    interceptor: &Interceptor,
) -> CoreResult<Value>
where
    Interceptor: CallInterceptor<AudioTranscriptionCall>,
{
    let PreparedAudioTranscriptionCall { context, request } =
        prepare_audio_transcription_call(request);
    CallRuntime::new(interceptor)
        .run::<AudioTranscriptionCall, _, _, _, _>(
            context,
            request,
            prepare_provider_request,
            execute_audio_transcription_provider_call,
        )
        .await
}

#[cfg(test)]
mod tests;
