mod host;

use std::sync::Arc;

use host::MessagesPythonHost;
use litellm_auth::AuthServices;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{call::OutputOf, machine::ChannelInterceptors};
use litellm_inference::{CallOptions, RouteError, caching::StreamCachable};
use litellm_inference_messages::{MessagesCall, MessagesRoute, route::Messages};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;

use super::{
    NativeCall,
    inference::{InferenceRoute, run_inference},
};

fn run_messages(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let host = MessagesPythonHost::new(call.view()?);
    run_inference::<MessagesRoute, _>(py, call, asynchronous, host)
}

#[pyfunction]
pub(crate) fn messages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, false)
}

#[pyfunction]
pub(crate) fn amessages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, true)
}

impl InferenceRoute for MessagesRoute {
    type Protocol = Messages;
    const OPERATION: LoggingOperation = LoggingOperation::Messages;
    const SYNC_CALL_TYPE: &'static str = "anthropic_messages";
    const ASYNC_CALL_TYPE: &'static str = "anthropic_messages";

    fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self::new(http, auth, secrets)
    }

    async fn call(
        self,
        request: MessagesCall,
        interceptors: ChannelInterceptors<Messages>,
        options: CallOptions,
    ) -> Result<OutputOf<Messages>, RouteError> {
        self.execute(request, &interceptors, options).await
    }
}

impl crate::cache::CachedStream for Messages {
    const TERMINAL_EVENT: Option<&'static str> = Some(<Self as StreamCachable>::TERMINAL_EVENT);

    fn replay(text: String) -> Option<(Self::StreamHead, Vec<Self::Chunk>)> {
        Some((
            litellm_inference_messages::route::MessagesStreamHead {
                headers: Vec::new(),
            },
            vec![text.into()],
        ))
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        <Self as StreamCachable>::bytes(chunk)
    }
}
