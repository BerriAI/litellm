mod common_utils;
mod constants;
mod handler;
mod prepare;
pub mod route;
mod types;

use futures_util::FutureExt;
use litellm_auth::AuthServices;
use litellm_host::interceptors::{ExecutionFacts, Interceptors, ResultSource};

use litellm_inference::{
    caching::{CachePlan, CacheSession},
    context::CallContext,
};
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

pub use litellm_inference::RouteError as Error;
pub use types::{MessagesCall, MessagesCallResponse, MessagesShaping, messages_body};

#[derive(Clone)]
pub struct MessagesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<Arc<dyn litellm_cache_response::ResponseCacheService>>,
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
            let cache = CachePlan::for_request(
                self.cache.as_ref(),
                context.cache.clone(),
                &call,
                context.model_group.as_deref(),
            )?;
            let prepared = prepare::prepare(call, self.secrets.as_ref()).await?;
            litellm_inference::diagnostic::provider(
                &prepared.body.model,
                prepared.provider.as_str(),
            );
            let mut request = self
                .prepare_outbound(prepared, cache, &context)
                .boxed()
                .await?;
            let session = CacheSession::<route::Messages>::open(request.cache.take()).await;
            let identity = request.identity.clone();
            let (output, source) = match CacheSession::replay(session.as_ref()).await {
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
            Ok(CacheSession::finish(session, output, &source).await)
        })
        .await
    }
}
