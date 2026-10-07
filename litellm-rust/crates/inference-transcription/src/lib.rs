pub mod types;
pub use litellm_inference::RouteError as Error;
use litellm_inference::call::{self, Failure};
mod constants;
mod handler;
mod prepare;
use std::sync::Arc;

pub use handler::execute_audio_transcription_provider_call;
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
pub use prepare::prepare_audio_transcription_provider_call;
use serde_json::Value;

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
    ) -> Result<Value, Failure<Error>> {
        litellm_inference::diagnostic::unary(async {
            let request = call::prepare(prepare_audio_transcription_provider_call(
                request,
                self.secrets.as_ref(),
            ))
            .await?;
            litellm_inference::diagnostic::provider(&request.model, &request.custom_llm_provider);
            let execute: futures_util::future::BoxFuture<'_, Result<Value, Failure<Error>>> =
                Box::pin(execute_audio_transcription_provider_call(
                    &self.http, &self.auth, request,
                ));
            execute.await
        })
        .await
    }
}
