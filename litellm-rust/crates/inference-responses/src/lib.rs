use litellm_host::observation::ObservationSender;
pub use litellm_inference::RouteError as Error;
use litellm_inference::caching::CachePlan;

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
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
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

    pub fn with_cache(
        self,
        cache: impl Into<Option<Arc<dyn litellm_cache_response::ResponseCacheService>>>,
    ) -> Self {
        Self {
            cache: cache.into(),
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
            model_group,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(
                call,
                cache_options,
                model_group.as_deref(),
                interceptors,
                observers.as_ref(),
            ),
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
        model_group: Option<&str>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_inference::diagnostic::call(async {
            self.run_provider(call, cache_options, model_group, interceptors, observers)
                .await
        })
        .await
    }

    async fn run_provider(
        &self,
        call: ResponsesCall,
        cache_options: Option<litellm_cache_response::CacheOptions>,
        model_group: Option<&str>,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        let cache = CachePlan::for_request(self.cache.as_ref(), cache_options, &call, model_group)?;
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
                cache,
                interceptors,
                observers,
            ));
        execute.await
    }
}
