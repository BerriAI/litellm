use litellm_host::observation::ObservationSender;
pub mod route;
pub mod types;
pub use crate::error::RouteError as Error;
mod common_utils;
pub(crate) mod handler;
mod prepare;
use litellm_llms_types::chat_completions::ChatCompletionsResponse;
use prepare::{prepare_provider_request, resolve_request};

use crate::chat_completions::types::ChatCompletionsRequest;
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

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
        options: impl Into<crate::CallOptions>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let crate::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run_call(
                request.into(),
                cache_options,
                interceptors,
                observers.as_ref(),
            ),
        )
        .await
    }

    async fn run(
        &self,
        request: ChatCompletionsRequest<'_>,
        cache_options: Option<litellm_cache_response::CacheOptions>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let resolved = resolve_request(request)?;
        let snapshot = self
            .secrets
            .resolve(&resolved.config.secret_names())
            .await?;
        let prepared = prepare_provider_request(resolved, snapshot)?;
        crate::diagnostic::provider(&prepared.model, &prepared.custom_llm_provider);
        let execute: futures_util::future::BoxFuture<'_, Result<ChatCompletionsResponse, Error>> =
            Box::pin(handler::execute(
                &self.http,
                &self.auth,
                prepared,
                self.cache.clone(),
                cache_options,
                interceptors,
                observers,
            ));
        execute.await
    }
}
