use litellm_host::observation::ObservationSender;
pub mod route;
pub mod types;
pub use litellm_inference::RouteError as Error;
mod common_utils;
pub mod constants;
pub(crate) mod handler;
mod provider_config;
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;
use types::{ChatCompletionsCall, ChatCompletionsRequest};

#[derive(Clone)]
pub struct ChatCompletionsRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl ChatCompletionsRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
            cache: None,
        }
    }

    pub fn with_cache(self, cache: litellm_cache_response::ScopedCache) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub async fn execute(
        &self,
        request: ChatCompletionsRequest<'_>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run(
                request.into(),
                cache_options,
                interceptors,
                observers.as_ref(),
            ),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %call.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(
        &self,
        call: ChatCompletionsCall,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ChatCompletionsResponse, Error> {
        litellm_inference::diagnostic::unary(async {
            let model = call.model.clone();
            let custom_llm_provider = call.custom_llm_provider.clone();
            let (provider, config) =
                provider_config::resolve_provider_config(&model, custom_llm_provider.as_deref())?;
            let call = common_utils::resolve_request(call, config)?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self.secrets.resolve(&config.secret_names()).await?;
            let execute: futures_util::future::BoxFuture<
                '_,
                Result<ChatCompletionsResponse, Error>,
            > = Box::pin(handler::execute(
                &self.http,
                &self.auth,
                config,
                provider,
                call,
                secrets,
                self.cache.clone(),
                cache_options,
                interceptors,
                observers,
            ));
            execute.await
        })
        .await
    }
}
