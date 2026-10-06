use litellm_host::observation::ObservationSender;
pub mod route;
pub mod types;
pub use litellm_inference::RouteError as Error;
mod common_utils;
pub mod constants;
pub(crate) mod handler;
mod prepare;

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;
use types::ChatCompletionsRequest;

use litellm_auth::AuthServices;
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use litellm_secrets::source::SecretSource;

use crate::caching::RouteCache;
use crate::chat_completions::types::{ChatCompletionsCall, ChatCompletionsRequest};
use prepare::{prepare_provider_request, resolve_request};

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

    pub fn with_cache(self, cache: impl Into<Option<litellm_cache_response::ScopedCache>>) -> Self {
        Self {
            cache: cache.into(),
            ..self
        }
    }

    pub async fn execute(
        &self,
        request: impl Into<ChatCompletionsCall>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            model_group,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run_call(
                request.into(),
                cache_options,
                model_group.as_deref(),
                interceptors,
                observers.as_ref(),
            ),
        )
        .await
    }

    async fn run(
        &self,
        request: ChatCompletionsRequest<'_>,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        model_group: Option<&str>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let cache = RouteCache::attach(self.cache.as_ref(), cache_options, &request, model_group)?;
        let resolved = resolve_request(request)?;
        let snapshot = self
            .secrets
            .resolve(&resolved.config.secret_names())
            .await?;
        let prepared = prepare_provider_request(resolved, snapshot)?;
        litellm_inference::diagnostic::provider(&prepared.model, &prepared.custom_llm_provider);
        let execute: futures_util::future::BoxFuture<'_, Result<ChatCompletionsResponse, Error>> =
            Box::pin(handler::execute(
                &self.http,
                &self.auth,
                prepared,
                cache,
                interceptors,
                observers,
            ));
        execute.await
    }
}
