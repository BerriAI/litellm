use litellm_host::observation::ObservationSender;
pub mod route;
pub mod types;
pub use litellm_inference::{Error, Unplanned};
mod common_utils;
pub mod constants;
pub(crate) mod handler;
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;

use litellm_auth::AuthServices;
use litellm_inference::provider::resolve_llm_provider;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;
use types::{ChatCompletionsCall, ChatCompletionsPlan};

#[derive(Clone)]
pub struct ChatCompletionsRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
}

impl ChatCompletionsRoute {
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

    /// Everything about a call that can be settled without a credential, a host or the network:
    /// the provider, its config, and whether that config translates the request. The only place
    /// this route declines.
    pub fn plan(call: impl Into<ChatCompletionsCall>) -> Result<ChatCompletionsPlan, Unplanned> {
        let call = call.into();
        let resolved = resolve_llm_provider(
            &call.model,
            call.custom_llm_provider.as_deref(),
            "chat completions",
        )?;
        let provider = resolved.provider;
        let model = resolved.model.to_owned();
        let config = common_utils::chat_completions_provider(provider)
            .ok_or_else(|| {
                litellm_inference::provider::no_native_config(
                    "chat completions",
                    <&str>::from(provider),
                )
            })?
            .config();
        let requested_model = call.model.clone();
        Ok(ChatCompletionsPlan {
            requested_model,
            model,
            provider,
            config,
            request: common_utils::resolve_request(call, config)?,
        })
    }

    pub async fn execute(
        &self,
        plan: ChatCompletionsPlan,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ChatCompletionsResponse, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run(plan, cache_options, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %plan.requested_model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(
        &self,
        plan: ChatCompletionsPlan,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ChatCompletionsResponse, Error> {
        litellm_inference::diagnostic::unary(async {
            let ChatCompletionsPlan {
                model,
                provider,
                config,
                request: call,
                ..
            } = plan;
            let provider = litellm_inference::provider::ResolvedProvider {
                model: &model,
                provider,
            };
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self.secrets.resolve(&config.secret_names()).await?;
            let execute: futures_util::future::BoxFuture<
                '_,
                Result<ChatCompletionsResponse, Error>,
            > = Box::pin(handler::execute(
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
