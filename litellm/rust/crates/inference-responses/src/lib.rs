use litellm_host::observation::ObservationSender;
pub use litellm_inference::{Error, Unplanned};

pub mod ids;
pub mod route;
pub mod types;
pub mod websocket;

mod handler;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, LlmProviders, get_custom_llm_provider,
};
use litellm_host::interceptors::Interceptors;
use litellm_host::interceptors::ProviderIdentity;
use litellm_llms::{
    base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
    openrouter::responses::transformation::OPENROUTER_RESPONSES_CONFIG,
};
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput, ResponsesPlan, ResponsesProvider};

fn responses_provider(provider: LlmProviders) -> Option<&'static dyn BaseResponsesApiConfig> {
    match provider {
        LlmProviders::Openai => Some(&OpenAiResponsesApiConfig),
        LlmProviders::Openrouter => Some(&OPENROUTER_RESPONSES_CONFIG),
        _ => None,
    }
}

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

    /// Resolves the provider and its native Responses config, which is everything about a call
    /// that can be settled before a credential or the network. The only place this route declines.
    pub fn plan(call: ResponsesCall) -> Result<ResponsesPlan, Unplanned> {
        let explicit = call
            .custom_llm_provider
            .as_deref()
            .filter(|provider| !provider.is_empty());
        let (provider_name, resolved_model) = match get_custom_llm_provider(&call.model, explicit) {
            Some(CustomLlmProvider {
                model,
                custom_llm_provider,
            }) => (custom_llm_provider, model),
            None => ("openai", call.model.as_str()),
        };
        if resolved_model.is_empty() {
            return Err(Error::InvalidProvider(call.model).into());
        }
        let not_native =
            || litellm_inference::provider::no_native_config("responses", provider_name);
        let provider = provider_name
            .parse::<LlmProviders>()
            .map_err(|_| not_native())?;
        let config = responses_provider(provider).ok_or_else(not_native)?;
        let wire_model = match provider {
            LlmProviders::Openai if resolved_model.contains('/') => {
                return Err(Error::InvalidProvider(call.model).into());
            }
            LlmProviders::Openai => resolved_model.into(),
            LlmProviders::Openrouter if explicit.is_some() && !resolved_model.contains('/') => {
                format!("openrouter/{resolved_model}")
            }
            LlmProviders::Openrouter => resolved_model.into(),
            _ => return Err(not_native().into()),
        };
        let provider = ResponsesProvider {
            identity: ProviderIdentity {
                model: wire_model,
                provider: provider_name.into(),
            },
            config,
        };
        Ok(ResponsesPlan { provider, call })
    }

    pub async fn execute(
        &self,
        plan: ResponsesPlan,
        interceptors: &impl Interceptors<Error>,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> Result<ResponsesOutput, Error> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(plan, cache_options, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "responses",
        model = %plan.call.model,
        provider,
        resolved_model,
        stream = plan.call.optional_params.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
        outcome
    ))]
    async fn run(
        &self,
        plan: ResponsesPlan,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_inference::diagnostic::call(async {
            let ResponsesPlan { provider, call } = plan;
            litellm_inference::diagnostic::provider(
                &provider.identity.model,
                &provider.identity.provider,
            );
            let secrets = self
                .secrets
                .resolve(
                    provider
                        .config
                        .secret_names(call.api_key.as_deref(), call.api_base.as_deref()),
                )
                .await?;
            let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
                Box::pin(handler::execute(
                    self,
                    provider,
                    call,
                    secrets,
                    cache_options,
                    interceptors,
                    observers,
                ));
            execute.await
        })
        .await
    }
}
