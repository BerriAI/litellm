pub mod types;
pub use litellm_inference::RouteError as Error;
mod constants;
mod handler;
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_inference::provider::{ResolvedProvider, resolve_llm_provider};
use litellm_llms::{
    base_llm::audio_transcription::transformation::BaseAudioTranscriptionConfig,
    bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG,
};
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use serde_json::Value;
use std::sync::Arc;

use crate::types::AudioTranscriptionRequest;

fn provider_config(provider: LlmProviders) -> Option<&'static dyn BaseAudioTranscriptionConfig> {
    match provider {
        LlmProviders::Bedrock => Some(&BEDROCK_AUDIO_TRANSCRIPTION_CONFIG),
        _ => None,
    }
}

fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<
    (
        ResolvedProvider<'a>,
        &'static dyn BaseAudioTranscriptionConfig,
    ),
    Error,
> {
    let provider = resolve_llm_provider(model, custom_llm_provider, "audio transcription")?;
    let config = provider_config(provider.provider)
        .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
    Ok((provider, config))
}

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
    pub async fn execute(&self, request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
        litellm_inference::diagnostic::unary(async {
            let (provider, config) = resolve_provider_config(
                request.model,
                request.custom_llm_provider,
            )?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self.secrets.resolve(&config.secret_names()).await?;
            let execute: futures_util::future::BoxFuture<'_, Result<Value, Error>> =
                Box::pin(handler::execute(
                    &self.http,
                    &self.auth,
                    config,
                    provider,
                    request,
                    secrets,
                ));
            execute.await
        })
        .await
    }
}
