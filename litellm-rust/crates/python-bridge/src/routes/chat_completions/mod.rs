mod host;

use std::sync::Arc;

use host::ChatCompletionsPythonHost;
use litellm_auth::AuthServices;
use litellm_cache_response::ScopedCache;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{
    call::{CallOutput, OutputOf},
    machine::ChannelInterceptors,
    protocol::Protocol,
};
use litellm_inference_chat::{
    ChatCompletionsRoute,
    route::ChatCompletions,
    types::{ChatCompletionsCall, ChatCompletionsRequest},
};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;

use super::{
    NativeCall,
    inference::{InferenceHost, InferenceRoute, run_inference},
};
use crate::cache::Cached;

fn run_chat_completions(
    py: Python<'_>,
    call: NativeCall<'_>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let host = InferenceHost::new(
        call.view()?,
        "litellm.rust_bridge.chat_completions.route_host",
        "chat_completions",
        if asynchronous {
            "acompletion"
        } else {
            "completion"
        },
    );
    run_inference::<ChatCompletionsRoute, _>(
        py,
        call,
        asynchronous,
        ChatCompletionsPythonHost(host),
    )
}

#[pyfunction]
pub(crate) fn completion(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_chat_completions(py, call, false)
}

#[pyfunction]
pub(crate) fn acompletion(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_chat_completions(py, call, true)
}

impl InferenceRoute for ChatCompletionsRoute {
    type Protocol = ChatCompletions;
    const OPERATION: LoggingOperation = LoggingOperation::Completion;

    fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self::new(http, auth, secrets)
    }

    fn with_cache(self, cache: ScopedCache) -> Self {
        self.with_cache(cache)
    }

    async fn execute(
        self,
        call: <ChatCompletions as Protocol>::Request,
        interceptors: ChannelInterceptors<Cached<ChatCompletions>>,
        options: litellm_inference::CallOptions,
    ) -> Result<OutputOf<ChatCompletions>, litellm_inference::RouteError> {
        let ChatCompletionsCall {
            model,
            messages,
            optional_params,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = call;
        let request = ChatCompletionsRequest {
            model: &model,
            messages,
            optional_params,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        };
        ChatCompletionsRoute::execute(&self, request, &interceptors, options)
            .await
            .map(CallOutput::Complete)
    }
}
