mod common_utils;
mod constants;
mod encoding;
mod handler;
mod host_answer;
mod pre_request;
mod reads;
pub mod route;
mod safeguards;
mod transport;
mod types;
mod workload_identity;

use futures_util::FutureExt;
use litellm_auth::AuthServices;
use litellm_host::interceptors::Interceptors;
use litellm_inference::context::CallContext;
use litellm_inference::provider::resolve_llm_provider;
use litellm_llms_types::recognized::Recognized;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

use crate::common_utils::messages_provider;

pub use common_utils::merge_messages_header_sources;
pub use encoding::DecodingError;
pub use host_answer::HostAnswer;
pub use litellm_inference::Error;
pub use pre_request::PreRequestHooks;
pub use safeguards::Safeguards;
pub use transport::{Delivered, Transport, TransportRequest, TransportResponse, UnencodableHeader};
pub use types::{
    MessagesBody, MessagesCall, MessagesCallResponse, MessagesSettings, MessagesShaping,
    litellm_params, messages_body,
};
pub use workload_identity::WorkloadIdentity;

#[derive(Clone)]
pub struct MessagesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
    cache: Option<litellm_cache_response::ScopedCache>,
    workload_identity: Option<Arc<dyn WorkloadIdentity>>,
    transport: Option<Arc<dyn Transport>>,
    pre_request_hooks: Option<Arc<dyn PreRequestHooks>>,
    safeguards: Option<Arc<dyn Safeguards>>,
    host_answer: Option<Arc<dyn HostAnswer>>,
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
            workload_identity: None,
            transport: None,
            pre_request_hooks: None,
            safeguards: None,
            host_answer: None,
        }
    }

    #[must_use]
    pub fn with_workload_identity(self, workload_identity: Arc<dyn WorkloadIdentity>) -> Self {
        Self {
            workload_identity: Some(workload_identity),
            ..self
        }
    }

    #[must_use]
    pub fn with_transport(self, transport: Arc<dyn Transport>) -> Self {
        Self {
            transport: Some(transport),
            ..self
        }
    }

    #[must_use]
    pub fn with_pre_request_hooks(self, hooks: Arc<dyn PreRequestHooks>) -> Self {
        Self {
            pre_request_hooks: Some(hooks),
            ..self
        }
    }

    #[must_use]
    pub fn with_safeguards(self, safeguards: Arc<dyn Safeguards>) -> Self {
        Self {
            safeguards: Some(safeguards),
            ..self
        }
    }

    #[must_use]
    pub fn with_host_answer(self, host_answer: Arc<dyn HostAnswer>) -> Self {
        Self {
            host_answer: Some(host_answer),
            ..self
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
        stream = call.body.params.streams(),
        outcome
    ))]
    async fn run(
        &self,
        call: MessagesCall,
        context: CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<MessagesCallResponse, Error> {
        litellm_inference::diagnostic::call(async {
            let requested_stream = call.body.params.streams();
            let client_streams = call.body.params.stream == Some(Recognized::Known(true));
            let call = match &self.pre_request_hooks {
                None => call,
                Some(hooks) => {
                    let (config, model, _) = Self::provider(&call)?;
                    if let Some(output) = handler::cached(self, config, &model, &context).await? {
                        return Ok(output);
                    }
                    hooks.run(call).await?
                }
            };
            let (config, model, provider) = Self::provider(&call)?;
            litellm_inference::diagnostic::provider(&model, provider);
            if self.pre_request_hooks.is_none()
                && let Some(output) = handler::cached(self, config, &model, &context).await?
            {
                return Ok(output);
            }
            if let Some(host_answer) = self
                .host_answer
                .as_deref()
                .filter(|_| call.shaping.settings.intercepted)
                && host_answer.intercept(requested_stream).await?
            {
                return handler::deliver_answer(self, config, model, requested_stream, &context)
                    .await;
            }
            if let Some(mock) = call.shaping.settings.mock_response {
                return Ok(
                    handler::mocked(self, config, model, &context, mock, requested_stream).await,
                );
            }
            if let Some(host_answer) = self
                .host_answer
                .as_deref()
                .filter(|_| call.shaping.settings.mcp_gateway)
            {
                let stream = call.body.params.streams();
                return handler::answered(self, host_answer, config, model, stream, &context).await;
            }
            let secrets = self.secrets.resolve(config.config().secret_names()).await?;
            handler::execute(self, config, model, call, secrets, client_streams, &context)
                .boxed()
                .await
        })
        .await
    }

    fn provider(
        call: &MessagesCall,
    ) -> Result<(common_utils::MessagesProvider, String, &'static str), Error> {
        let provider = resolve_llm_provider(
            &call.body.model,
            call.custom_llm_provider.as_deref(),
            "messages",
        )?;
        let config = messages_provider(provider.provider, provider.model)
            .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
        Ok((
            config,
            provider.model.to_owned(),
            <&'static str>::from(provider.provider),
        ))
    }
}
