use std::sync::Arc;

use serde_json::Value;

use super::types::{AnthropicMessagesResponse, MessagesRequest};
use crate::call_lifecycle::provider::{
    CompletedCall, CompletedRoute, ProviderHooks, ProviderOptions,
};
use crate::call_lifecycle::workflow::WorkflowFuture;

pub struct OwnedMessagesRequest {
    pub options: ProviderOptions,
    pub body: Value,
}

impl From<MessagesRequest<'_>> for OwnedMessagesRequest {
    fn from(request: MessagesRequest<'_>) -> Self {
        Self {
            options: ProviderOptions {
                model: request.model.to_owned(),
                litellm_call_id: None,
                api_key: request.api_key.map(str::to_owned),
                api_base: request.api_base.map(str::to_owned),
                custom_llm_provider: request.custom_llm_provider.map(str::to_owned),
                extra_headers: request.extra_headers,
                timeout: request.timeout,
            },
            body: request.body,
        }
    }
}

pub struct MessagesRoute;
pub type MessagesCall = CompletedCall<MessagesRoute>;

impl CompletedRoute for MessagesRoute {
    type Admission = crate::call_lifecycle::admission::Inspection<super::MessagesAdmission>;
    type Request = OwnedMessagesRequest;
    type Response = AnthropicMessagesResponse;

    fn admit(
        admission: Self::Admission,
    ) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
        super::admit(admission)
    }

    fn run(
        request: Self::Request,
        hooks: Arc<dyn ProviderHooks>,
    ) -> WorkflowFuture<Self::Response> {
        Box::pin(async move {
            let options = request.options;
            let request = MessagesRequest {
                model: &options.model,
                api_key: options.api_key.as_deref(),
                api_base: options.api_base.as_deref(),
                custom_llm_provider: options.custom_llm_provider.as_deref(),
                extra_headers: options.extra_headers,
                timeout: options.timeout,
                body: request.body,
            };
            super::handler::execute_messages_provider_call(request, hooks.as_ref()).await
        })
    }

    fn context(request: &Self::Request) -> crate::call_lifecycle::CallLifecycleContext {
        request.options.lifecycle_context("anthropic_messages")
    }
}
