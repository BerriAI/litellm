use litellm_host::observation::ObservationSender;
use std::convert::Infallible;

use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;

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
                self.run_call(call, cache_options, &interceptors, observers.as_ref())
                    .await
                    .map(CallOutput::Complete)
            },
        )
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %call.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    pub(super) async fn run_call(
        &self,
        call: ChatCompletionsCall,
        cache_options: Option<litellm_cache_response::CachePolicy>,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ChatCompletionsResponse, Error> {
        litellm_inference::diagnostic::unary(async {
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
            self.run(request, cache_options, interceptors, observers)
                .await
        })
        .await
    }
}

impl litellm_inference::caching::Cachable for ChatCompletions {
    const SURFACE: &'static str = "chat_completions";
}
