use litellm_host::observation::ObservationSender;
pub use litellm_inference::RouteError as Error;

pub mod route;
pub mod types;
pub mod websocket;

mod handler;
mod provider_config;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_host::interceptors::Interceptors;
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

use crate::provider_config::resolve_provider_config;

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
            let model = call.model.clone();
            let custom_llm_provider = call.custom_llm_provider.clone();
            let (provider, config) =
                resolve_provider_config(&model, custom_llm_provider.as_deref())?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self
                .secrets
                .resolve(config.secret_names(call.api_key.as_deref(), call.api_base.as_deref()))
                .await?;
            let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
                Box::pin(handler::execute(
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
