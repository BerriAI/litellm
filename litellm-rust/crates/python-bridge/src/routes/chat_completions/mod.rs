mod host;

use std::sync::Arc;

use host::ChatCompletionsPythonHost;
use litellm_auth::AuthServices;
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
        "litellm.rust_bridge.host.routes.chat_completions",
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

    async fn call(
        self,
        request: ChatCompletionsCall,
        interceptors: ChannelInterceptors<ChatCompletions>,
        options: CallOptions,
    ) -> Result<OutputOf<ChatCompletions>, RouteError> {
        self.execute_call(request, &interceptors, options)
            .await
            .map(CallOutput::Complete)
    }
}

impl crate::cache::CachedStream for ChatCompletions {
    const TERMINAL_EVENT: Option<&'static str> = None;

    fn replay(_: String) -> Option<(Self::StreamHead, Vec<Self::Chunk>)> {
        None
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        match *chunk {}
    }
}
