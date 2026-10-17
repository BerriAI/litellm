//! Helpers the passthrough tests of several providers share.

use std::sync::{Arc, Mutex};

use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, WireRequest,
};
use litellm_inference::{CallOptions, context::CallContext};
use rstest::fixture;

use crate::{
    ChatCompletionsRoute, Error,
    openai::{OpenAiChatCall, OpenAiChatOutput},
};

#[derive(Default)]
pub(crate) struct Recording {
    pub(crate) contexts: Mutex<Vec<RequestContext>>,
    pub(crate) raw: Mutex<Vec<String>>,
    pub(crate) results: Mutex<Vec<ExecutionFacts>>,
}

impl Interceptors<Error> for Recording {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        self.contexts.lock().unwrap().push(context);
        Ok(wire)
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
        self.raw.lock().unwrap().push(raw.body);
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        self.results.lock().unwrap().push(facts);
        Ok(())
    }
}

#[fixture]
pub(crate) fn route() -> ChatCompletionsRoute {
    ChatCompletionsRoute::new(
        litellm_http::Client::plain_for_test(),
        Arc::new(litellm_auth::AuthServices::default()),
        litellm_inference_testing::no_secrets(),
    )
}

pub(crate) async fn run(
    route: &ChatCompletionsRoute,
    call: OpenAiChatCall,
    interceptors: &Recording,
) -> Result<OpenAiChatOutput, Error> {
    route
        .run_openai(call, CallContext::new(interceptors, CallOptions::default()))
        .await
}
