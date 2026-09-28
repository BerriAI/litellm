pub use crate::error::RouteError as Error;
use litellm_host::observation::ObservationSender;
pub mod websocket;

mod handler;
mod prepare;
pub mod route;
pub mod types;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_host::interceptors::Interceptors;
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

#[derive(Clone)]
pub struct ResponsesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl ResponsesRoute {
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
        call: ResponsesCall,
        interceptors: &impl Interceptors<Error>,
        options: impl Into<crate::CallOptions>,
    ) -> Result<ResponsesOutput, Error> {
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
        route = "responses",
        model = %call.model,
        provider,
        resolved_model,
        stream = call.optional_params.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
        outcome
    ))]
    async fn run(
        &self,
        call: ResponsesCall,
        cache_options: Option<litellm_cache_response::CacheOptions>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        crate::diagnostic::call(async {
            crate::caching::execute_streaming::<route::Responses, _, _>(
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
        call: ResponsesCall,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        let request = prepare::prepare(call, self.secrets.as_ref()).await?;
        crate::diagnostic::provider(&request.context.model, &request.context.custom_llm_provider);
        let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> = Box::pin(
            handler::execute(&self.http, &self.auth, request, interceptors, observers),
        );
        execute.await
    }
}
