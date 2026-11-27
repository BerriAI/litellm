use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;

use super::{ChatCompletionsRoute, Error, types::ChatCompletionsCall};

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
    pub fn machine(
        self,
        call: ChatCompletionsCall,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> HostedMachine<ChatCompletions> {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                self.run(call, cache_options, &interceptors, observers.as_ref())
                    .await
                    .map(CallOutput::Complete)
            },
        )
    }
}

impl litellm_inference::caching::Cachable for ChatCompletions {
    const SURFACE: &'static str = "chat_completions";
}
