mod error;
mod handler;
mod route;
mod types;

use std::sync::{Arc, OnceLock};

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
mod local;

pub use litellm_inference::Error;
pub use local::LocalTokenizerRegistry;
pub use route::{Tokenize, TokenizeCall, TokenizeMachine};
pub use types::{TokenizeBackend, TokenizeRequest};

static DEFAULT_LOCAL_REGISTRY: OnceLock<Arc<LocalTokenizerRegistry>> = OnceLock::new();

#[derive(Clone)]
pub struct TokenizeRoute {
    http: Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    local_registry: Arc<LocalTokenizerRegistry>,
}

impl TokenizeRoute {
    pub fn new(http: Client, auth: Arc<AuthServices>, secrets: Arc<dyn SecretSource>) -> Self {
        Self {
            http,
            auth,
            secrets,
            local_registry: Arc::clone(
                DEFAULT_LOCAL_REGISTRY.get_or_init(|| Arc::new(LocalTokenizerRegistry::new())),
            ),
        }
    }

    pub fn with_local_registry(mut self, registry: Arc<LocalTokenizerRegistry>) -> Self {
        self.local_registry = registry;
        self
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
            if matches!(&call.backend, TokenizeBackend::Local { .. }) {
                return handler::execute_local(self, call, interceptors, observers).await;
            }
            let provider =
                resolve_llm_provider(&call.model, call.custom_llm_provider.as_deref(), "tokenize")?;
            let config: &dyn BaseTokenizeConfig = match provider.provider {
                LlmProviders::HostedVllm | LlmProviders::Vllm => &HostedVllmTokenizeConfig,
                LlmProviders::Watsonx => &WatsonxTokenizeConfig,
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
                return Err(Error::Unsupported("add_special_tokens"));
            }
            if !config
                .tokenize_capabilities()
                .supports(call.include.output())
            {
                return Err(Error::Unsupported("requested tokenize output"));
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
