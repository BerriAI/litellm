mod handler;
mod route;
mod types;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_host::{interceptors::Interceptors, observation::ObservationSender};
use litellm_http::Client;
use litellm_inference::provider::resolve_llm_provider;
use litellm_llms::{
    base_llm::tokenize::transformation::BaseTokenizeConfig,
    hosted_vllm::tokenize::transformation::HostedVllmTokenizeConfig,
    watsonx::tokenize::transformation::WatsonxTokenizeConfig,
};
use litellm_llms_types::formats::tokenize::TokenizeResponse;
use litellm_secrets::source::SecretSource;

pub use litellm_inference::Error;
pub use route::{Tokenize, TokenizeCall, TokenizeMachine};
pub use types::TokenizeRequest;

#[derive(Clone)]
pub struct TokenizeRoute {
    http: Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl TokenizeRoute {
    pub fn new(http: Client, auth: Arc<AuthServices>, secrets: Arc<dyn SecretSource>) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    pub async fn execute(
        &self,
        request: TokenizeRequest,
        interceptors: &impl Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<TokenizeResponse, Error> {
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run(request, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "tokenize", model = %call.model, resolved_model, provider,
        stream = false, outcome
    ))]
    async fn run(
        &self,
        call: TokenizeRequest,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<TokenizeResponse, Error> {
        litellm_inference::diagnostic::unary(async {
            let provider =
                resolve_llm_provider(&call.model, call.custom_llm_provider.as_deref(), "tokenize")?;
            let config: &dyn BaseTokenizeConfig = match provider.provider {
                LlmProviders::HostedVllm | LlmProviders::Vllm => &HostedVllmTokenizeConfig,
                LlmProviders::Watsonx => &WatsonxTokenizeConfig,
                _ => return Err(Error::Unsupported("tokenize provider")),
            };
            litellm_inference::diagnostic::provider(provider.model, provider.provider.into());
            let secrets = self.secrets.resolve(&config.secret_names()).await?;
            let execute: futures_util::future::BoxFuture<'_, Result<TokenizeResponse, Error>> =
                Box::pin(handler::execute(
                    self,
                    &call,
                    config,
                    provider,
                    secrets,
                    interceptors,
                    observers,
                ));
            execute.await
        })
        .await
    }
}
