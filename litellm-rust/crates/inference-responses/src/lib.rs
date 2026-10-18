use litellm_host::observation::ObservationSender;
pub use litellm_inference::RouteError as Error;

pub mod route;
pub mod types;
pub mod websocket;

mod handler;
mod prepare;

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
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ResponsesOutput, Error> {
        let litellm_inference::CallOptions {
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
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_inference::diagnostic::call(async {
            self.run_provider(call, cache_options, interceptors, observers)
                .await
        })
        .await
    }

    async fn run_provider(
        &self,
        call: ResponsesCall,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        let request = prepare::prepare(call, self.secrets.as_ref()).await?;
        litellm_inference::diagnostic::provider(
            &request.context.model,
            &request.context.custom_llm_provider,
        );
        let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
            Box::pin(handler::execute(
                &self.http,
                &self.auth,
                request,
                self.cache.clone(),
                cache_options,
                interceptors,
                observers,
            ));
        execute.await
    }
}
