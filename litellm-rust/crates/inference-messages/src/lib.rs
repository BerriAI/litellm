mod common_utils;
mod constants;
mod handler;
mod prepare;
pub mod route;
mod types;

use futures_util::FutureExt;
use litellm_auth::AuthServices;
use litellm_host::interceptors::{ExecutionFacts, Interceptors, ResultSource};

use litellm_inference::{caching::CallCache, context::CallContext};
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

use crate::caching::CacheKeyProjection;
pub use crate::error::RouteError as Error;
pub use types::{MessagesCall, MessagesCallResponse, MessagesShaping, messages_body};

#[derive(Clone)]
pub struct MessagesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl MessagesRoute {
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

    #[must_use]
    pub fn with_cache(self, cache: litellm_cache_response::ScopedCache) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }

    pub async fn execute(
        &self,
        call: MessagesCall,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<MessagesCallResponse, Error> {
        let context = CallContext::new(interceptors, options.into());
        litellm_host::lifecycle::observe_call(context.observers.clone(), self.run(call, context))
            .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "messages",
        model = %call.body.model,
        provider,
        resolved_model,
        stream = call.body.params.stream == Some(true),
        outcome
    ))]
    async fn run(
        &self,
        call: MessagesCall,
        context: CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<MessagesCallResponse, Error> {
        crate::diagnostic::call(async {
            let cache_input = self
                .cache
                .as_ref()
                .map(|_| call.cache_key_input())
                .transpose()?
                .unwrap_or_default();
            let prepared = prepare::prepare(call, self.secrets.as_ref()).await?;
            litellm_inference::diagnostic::provider(
                &prepared.body.model,
                prepared.provider.as_str(),
            );
            let request = self.prepare_outbound(prepared, &context).boxed().await?;
            let cache = CallCache::<route::Messages>::prepare(
                self.cache.as_ref().filter(|_| request.cacheable()),
                context.cache,
                crate::caching::CacheRequest::from_logical(
                    request.identity.clone(),
                    cache_input,
                    request.original.as_ref(),
                    &request.wire,
                )
                .input,
            )
            .await;
            let identity = request.identity.clone();
            let (output, source) = match cache.lookup().await {
                Some(hit) => hit,
                None => (
                    self.call_provider(request, &context).await?,
                    ResultSource::Provider,
                ),
            };
            context
                .result_ready(ExecutionFacts {
                    provider: identity,
                    source: source.clone(),
                })
                .await?;
            Ok(cache.finish(output, &source).await)
        })
        .await
    }
}
