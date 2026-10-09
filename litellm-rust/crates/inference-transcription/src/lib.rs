pub mod types;
pub use litellm_inference::RouteError as Error;
mod constants;
mod handler;
mod provider_config;

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use serde_json::Value;
use std::sync::Arc;

use crate::{provider_config::resolve_provider_config, types::AudioTranscriptionRequest};

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

    pub async fn execute(&self, request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
        self.run(request).await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "audio_transcription",
        model = request.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(&self, request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
        litellm_inference::diagnostic::unary(async {
            let (provider, config) =
                resolve_provider_config(request.model, request.custom_llm_provider)?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self.secrets.resolve(&config.secret_names()).await?;
            let execute: futures_util::future::BoxFuture<'_, Result<Value, Error>> = Box::pin(
                handler::execute(&self.http, &self.auth, config, provider, request, secrets),
            );
            execute.await
        })
        .await
    }
}
