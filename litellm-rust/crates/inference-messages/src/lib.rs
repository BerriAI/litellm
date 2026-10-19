mod common_utils;
mod constants;
mod handler;
mod prepare;
pub mod route;
mod types;

use futures_util::FutureExt;
use litellm_auth::AuthServices;
use litellm_host::interceptors::{Cost, ExecutionFacts, Interceptors, ResultSource};

use litellm_inference::{caching::CallCache, context::CallContext};
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

pub use litellm_inference::RouteError as Error;
pub use types::{
    MessagesCall, MessagesCallResponse, MessagesSettings, MessagesShaping, litellm_params,
    messages_body,
};

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
        litellm_inference::diagnostic::call(async {
            let prepared = prepare::prepare(call, self.secrets.as_ref()).await?;
            litellm_inference::diagnostic::provider(
                &prepared.body.model,
                prepared.provider.as_str(),
            );
            let request = self.prepare_outbound(prepared).boxed().await?;
            let early_cache = self
                .cache
                .as_ref()
                .filter(|cache| request.cacheable() && !cache.service.requires_wire_request())
                .map(|cache| {
                    CallCache::<route::Messages>::from_wire(
                        Some(cache),
                        context.cache,
                        &request.identity,
                        &request.wire,
                    )
                });
            let early_hit = match &early_cache {
                Some(cache) => cache.lookup().await,
                None => None,
            };
            let (request, stream) = self.intercept_outbound(request, &context).await?;
            let (cache, hit) = match early_cache {
                Some(cache) => (cache, early_hit),
                None => {
                    let cache = CallCache::<route::Messages>::from_wire(
                        self.cache.as_ref().filter(|_| request.cacheable()),
                        context.cache,
                        &request.identity,
                        &request.wire,
                    );
                    let hit = cache.lookup().await;
                    (cache, hit)
                }
            };
            let identity = request.identity.clone();
            let config = request.provider.config();
            let gateway_injected_cache_control = request.gateway_injected_cache_control;
            let (output, source) = match hit {
                Some(hit) => hit,
                None => (
                    self.call_provider(request, stream, &context).await?,
                    ResultSource::Provider,
                ),
            };
            let reported = match &output {
                MessagesCallResponse::Complete(response) => config.reported_cost(response),
                MessagesCallResponse::Stream { .. } => None,
            };
            context
                .result_ready(ExecutionFacts {
                    provider: identity,
                    cost: Cost::settle(&source, reported),
                    gateway_injected_cache_control: gateway_injected_cache_control
                        && source == ResultSource::Provider,
                    source: source.clone(),
                })
                .await?;
            let output = litellm_inference::caching::provider_result::<route::Messages>(
                output,
                &source,
                context.interceptors,
            )
            .await?;
            Ok(cache.finish(output, &source).await)
        })
        .await
    }
}
