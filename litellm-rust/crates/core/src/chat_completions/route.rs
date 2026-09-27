use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_types::utils::ChatCompletionsResponse;

use super::{
    ChatCompletionsRoute, Error,
    types::{ChatCompletionsCall, ChatCompletionsRequest},
};

pub struct ChatCompletions;

impl Protocol for ChatCompletions {
    type Response = ChatCompletionsResponse;
    type Error = Error;
    type Request = ChatCompletionsCall;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

impl ChatCompletionsRoute {
    pub fn machine(self, call: ChatCompletionsCall) -> HostedMachine<ChatCompletions> {
        hosted_call(
            call,
            move |call: ChatCompletionsCall, _, hooks| async move {
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
                self.run(request, &hooks).await.map(CallOutput::Complete)
            },
        )
    }
}
