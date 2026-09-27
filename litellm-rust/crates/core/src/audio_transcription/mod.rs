pub mod types;
pub use crate::error::RouteError as Error;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;

use crate::audio_transcription::types::AudioTranscriptionRequest;

impl crate::CoreClient {
    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "audio_transcription",
        model = request.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    pub async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest<'_>,
    ) -> Result<Value, Error> {
        crate::diagnostic::unary(async {
            let request = prepare_audio_transcription_provider_call(request)?;
            crate::diagnostic::provider(&request.model, &request.custom_llm_provider);
            let http = self.provider_http()?;
            let execute: futures_util::future::BoxFuture<'_, Result<Value, Error>> = Box::pin(
                execute_audio_transcription_provider_call(&http, &self.resources().auth, request),
            );
            execute.await
        })
        .await
    }
}
