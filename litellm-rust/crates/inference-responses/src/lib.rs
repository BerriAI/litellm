use litellm_host::observation::ObservationSender;
pub use litellm_inference::Error;

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
use types::{ResponsesCall, ResponsesOutput, ResponsesProvider};

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
            let explicit = call
                .custom_llm_provider
                .as_deref()
                .filter(|provider| !provider.is_empty());
            let (provider_name, resolved_model) =
                match get_custom_llm_provider(&call.model, explicit) {
                    Some(CustomLlmProvider {
                        model,
                        custom_llm_provider,
                    }) => (custom_llm_provider, model),
                    None => ("openai", call.model.as_str()),
                };
            if resolved_model.is_empty() {
                return Err(Error::InvalidProvider(call.model));
            }
            let provider = provider_name
                .parse::<LlmProviders>()
                .map_err(|_| Error::Unsupported("native HTTP responses provider"))?;
            let config = responses_provider(provider)
                .ok_or(Error::Unsupported("native HTTP responses provider"))?;
            let wire_model = match provider {
                LlmProviders::Openai if resolved_model.contains('/') => {
                    return Err(Error::InvalidProvider(call.model));
                }
                LlmProviders::Openai => resolved_model.into(),
                LlmProviders::Openrouter if explicit.is_some() && !resolved_model.contains('/') => {
                    format!("openrouter/{resolved_model}")
                }
                LlmProviders::Openrouter => resolved_model.into(),
                _ => return Err(Error::Unsupported("native HTTP responses provider")),
            };
            let provider = ResponsesProvider {
                identity: ProviderIdentity {
                    model: wire_model,
                    provider: provider_name.into(),
                },
                config,
            };
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
