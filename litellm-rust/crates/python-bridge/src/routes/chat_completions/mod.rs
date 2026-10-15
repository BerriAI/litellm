mod host;

use std::sync::Arc;

use host::ChatCompletionsPythonHost;
use litellm_auth::AuthServices;
use litellm_cache_response::ScopedCache;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{
    call::{CallOutput, OutputOf},
    machine::ChannelInterceptors,
};
use litellm_inference::{CallOptions, RouteError};
use litellm_inference_chat::{
    ChatCompletionsRoute, route::ChatCompletions, types::ChatCompletionsCall,
};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;

use crate::cache::Cached;

use super::{
    NativeCall,
    inference::{InferenceHost, InferenceRoute, run_inference},
};

fn run_chat_completions(
    py: Python<'_>,
    call: NativeCall<'_>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let host = InferenceHost::new(
        call.view()?,
        "litellm.rust_bridge.chat_completions.route_host",
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
    const SYNC_CALL_TYPE: &'static str = "completion";
    const ASYNC_CALL_TYPE: &'static str = "acompletion";

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

    async fn call(
        self,
        request: ChatCompletionsCall,
        interceptors: ChannelInterceptors<Cached<ChatCompletions>>,
        options: CallOptions,
    ) -> Result<OutputOf<ChatCompletions>, RouteError> {
        self.execute_call(request, &interceptors, options)
            .await
            .map(CallOutput::Complete)
    }
}
