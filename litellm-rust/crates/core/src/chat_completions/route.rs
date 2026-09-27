use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_types::utils::ChatCompletionsResponse;

use super::{
    Error,
    types::{ChatCompletionsCall, ChatCompletionsRequest},
};

pub struct ChatCompletions;

impl Protocol for ChatCompletions {
    type Response = ChatCompletionsResponse;
    type Error = Error;
    type Projection = ChatCompletionsCall;
    type Op = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

pub fn chat_completions_machine(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
) -> Result<HostedMachine<ChatCompletions>, litellm_http::Error> {
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    let auth = resources.auth.clone();
    Ok(hosted_call(
        move |call: ChatCompletionsCall, host| async move {
            let request = ChatCompletionsRequest {
                model: &call.model,
                messages: call.messages,
                optional_params: call.optional_params,
                api_key: call.api_key.as_deref(),
                api_base: call.api_base.as_deref(),
                custom_llm_provider: call.custom_llm_provider.as_deref(),
                extra_headers: call.extra_headers,
                timeout: call.timeout,
            };
            super::execute(&http, &auth, request, &host)
                .await
                .map(CallOutput::Complete)
        },
    ))
}
