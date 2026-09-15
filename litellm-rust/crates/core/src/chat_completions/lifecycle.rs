use std::sync::Arc;

use serde_json::Value;

use super::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use crate::call_lifecycle::provider::{
    CompletedCall, CompletedRoute, ProviderHooks, ProviderOptions,
};
use crate::call_lifecycle::workflow::WorkflowFuture;

pub struct OwnedChatCompletionsRequest {
    pub options: ProviderOptions,
    pub messages: Value,
    pub optional_params: serde_json::Map<String, Value>,
}

impl From<ChatCompletionsRequest<'_>> for OwnedChatCompletionsRequest {
    fn from(request: ChatCompletionsRequest<'_>) -> Self {
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
            messages: request.messages,
            optional_params: request.optional_params,
        }
    }
}

pub struct ChatCompletionsRoute;
pub type ChatCompletionsCall = CompletedCall<ChatCompletionsRoute>;

impl CompletedRoute for ChatCompletionsRoute {
    type Admission = crate::call_lifecycle::admission::Inspection<super::ChatCompletionsAdmission>;
    type Request = OwnedChatCompletionsRequest;
    type Response = ChatCompletionsResponse;

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
            let request = ChatCompletionsRequest {
                model: &options.model,
                api_key: options.api_key.as_deref(),
                api_base: options.api_base.as_deref(),
                custom_llm_provider: options.custom_llm_provider.as_deref(),
                extra_headers: options.extra_headers,
                timeout: options.timeout,
                messages: request.messages,
                optional_params: request.optional_params,
            };
            super::handler::execute_chat_completions_provider_call(
                super::prepare::resolve_request(request)?,
                hooks.as_ref(),
            )
            .await
        })
    }

    fn context(request: &Self::Request) -> crate::call_lifecycle::CallLifecycleContext {
        request.options.lifecycle_context("completion")
    }
}
