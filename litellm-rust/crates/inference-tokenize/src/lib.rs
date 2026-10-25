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
};
use litellm_llms_types::formats::tokenize::TokenizeResponse;
use litellm_secrets::source::SecretSource;

pub use litellm_inference::Error;
pub use route::{Tokenize, TokenizeCall, TokenizeMachine};
pub use types::TokenizeRequest;

#[derive(Clone)]
pub struct TokenizeRoute {
    http: Client,
    credential: litellm_http::CredentialClient,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl TokenizeRoute {
    pub fn new(
        http: Client,
        credential: litellm_http::CredentialClient,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            credential,
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
            if call.model.is_empty() {
                return Err(Error::InvalidRequest("model must not be empty".into()));
            }
            let provider =
                resolve_llm_provider(&call.model, call.custom_llm_provider.as_deref(), "tokenize")?;
            let config: &dyn BaseTokenizeConfig = match provider.provider {
                LlmProviders::HostedVllm => &HostedVllmTokenizeConfig,
                _ => return Err(Error::Unsupported("tokenize provider")),
            };
            if call.add_special_tokens.is_some()
                && call
                    .optional_params
                    .contains_key("add_special_tokens")
            {
                return Err(Error::InvalidRequest(
                    "add_special_tokens cannot be specified in both the top-level request and optional_params"
                        .into(),
                ));
            }
            if call.add_special_tokens.is_some()
                && !config
                    .get_supported_tokenize_params()
                    .contains(&"add_special_tokens")
            {
                return Err(Error::InvalidRequest(
                    "add_special_tokens is not supported by the selected provider".into(),
                ));
            }
            if !config
                .tokenize_capabilities()
                .supports(call.output())
            {
                return Err(Error::InvalidRequest(
                    "include requests a token representation not supported by the selected provider"
                        .into(),
                ));
            }
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
