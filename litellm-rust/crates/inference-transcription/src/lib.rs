pub mod types;
pub use litellm_inference::RouteError as Error;
mod constants;
mod handler;
mod prepare;
pub use handler::execute_audio_transcription_provider_call;
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;
use std::sync::Arc;

use crate::types::AudioTranscriptionRequest;

#[derive(Clone)]
pub struct AudioTranscriptionRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl AudioTranscriptionRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "audio_transcription",
        model = request.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    pub async fn execute(
        &self,
        request: AudioTranscriptionRequest<'_>,
    ) -> Result<http::Response<Value>, Error> {
        litellm_inference::diagnostic::unary(async {
            let request =
                prepare_audio_transcription_provider_call(request, self.secrets.as_ref()).await?;
            litellm_inference::diagnostic::provider(&request.model, &request.custom_llm_provider);
            let execute: futures_util::future::BoxFuture<'_, Result<http::Response<Value>, Error>> =
                Box::pin(execute_audio_transcription_provider_call(
                    &self.http, &self.auth, request,
                ));
            execute.await
        })
        .await
    }
}
