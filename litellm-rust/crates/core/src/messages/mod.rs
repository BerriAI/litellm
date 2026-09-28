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

#[must_use]
#[derive(Clone, Default)]
pub struct MessagesRouteBuilder<Http = (), Auth = (), Secrets = ()> {
    http: Http,
    auth: Auth,
    secrets: Secrets,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl<Http, Auth, Secrets> MessagesRouteBuilder<Http, Auth, Secrets> {
    pub fn with_http(
        self,
        http: litellm_http::Client,
    ) -> MessagesRouteBuilder<litellm_http::Client, Auth, Secrets> {
        MessagesRouteBuilder {
            http,
            auth: self.auth,
            secrets: self.secrets,
            cache: self.cache,
        }
    }

    pub fn with_auth(
        self,
        auth: Arc<AuthServices>,
    ) -> MessagesRouteBuilder<Http, Arc<AuthServices>, Secrets> {
        MessagesRouteBuilder {
            http: self.http,
            auth,
            secrets: self.secrets,
            cache: self.cache,
        }
    }

    pub fn with_secrets(
        self,
        secrets: Arc<dyn SecretSource>,
    ) -> MessagesRouteBuilder<Http, Auth, Arc<dyn SecretSource>> {
        MessagesRouteBuilder {
            http: self.http,
            auth: self.auth,
            secrets,
            cache: self.cache,
        }
    }

    pub fn with_cache(self, cache: litellm_cache_response::ScopedCache) -> Self {
        Self {
            cache: Some(cache),
            ..self
        }
    }
}

impl MessagesRouteBuilder<litellm_http::Client, Arc<AuthServices>, Arc<dyn SecretSource>> {
    pub fn build(self) -> MessagesRoute {
        MessagesRoute {
            http: self.http,
            auth: self.auth,
            secrets: self.secrets,
            cache: self.cache,
        }
    }
}

impl MessagesRoute {
    pub fn builder() -> MessagesRouteBuilder {
        MessagesRouteBuilder::default()
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
