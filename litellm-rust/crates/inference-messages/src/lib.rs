mod common_utils;
mod constants;
mod handler;
mod provider_config;
pub mod route;
mod types;

use futures_util::FutureExt;
use litellm_auth::AuthServices;
use litellm_host::interceptors::Interceptors;
use litellm_inference::context::CallContext;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

use crate::provider_config::resolve_provider_config;

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
            let model = call.body.model.clone();
            let custom_llm_provider = call.custom_llm_provider.clone();
            let (provider, config) =
                resolve_provider_config(&model, custom_llm_provider.as_deref())?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self.secrets.resolve(config.config().secret_names()).await?;
            handler::execute(self, config, provider, call, secrets, &context)
                .boxed()
                .await
        })
        .await
    }
}
