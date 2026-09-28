use litellm_host::observation::ObservationSender;
mod common_utils;
mod handler;
mod prepare;
pub mod route;
mod types;

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

pub use crate::error::RouteError as Error;
pub use types::{MessagesCall, MessagesResponse, MessagesShaping, messages_body};

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
        options: impl Into<crate::CallOptions>,
    ) -> Result<MessagesResponse, Error> {
        let crate::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(call, cache_options, interceptors, observers.as_ref()),
        )
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
        cache_options: Option<litellm_cache_response::CacheOptions>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<MessagesResponse, Error> {
        crate::diagnostic::call(async {
            crate::caching::execute_streaming::<route::Messages, _, _>(
                call,
                self.cache.as_ref().map(|cache| cache.service.clone()),
                self.cache
                    .as_ref()
                    .map(|cache| cache.options(cache_options)),
                interceptors,
                observers,
                |call| self.run_provider(call, interceptors, observers),
            )
            .await
        })
        .await
    }

    async fn run_provider(
        &self,
        call: MessagesCall,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<MessagesResponse, Error> {
        let request = prepare::prepare(call, self.secrets.as_ref()).await?;
        crate::diagnostic::provider(&request.body.model, request.provider.as_str());
        let execute: futures_util::future::BoxFuture<'_, Result<MessagesResponse, Error>> =
            Box::pin(handler::execute(
                &self.http,
                &self.auth,
                request,
                interceptors,
                observers,
            ));
        execute.await
    }
}
