mod host;
mod websocket;

use std::sync::Arc;

use host::ResponsesPythonHost;
use litellm_auth::AuthServices;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{call::OutputOf, machine::ChannelInterceptors};
use litellm_inference::{CallOptions, RouteError, caching::StreamCachable};
use litellm_inference_responses::{ResponsesRoute, route::Responses, types::ResponsesCall};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;
pub(crate) use websocket::ResponsesWebSocketConnection;

use super::{
    NativeCall,
    inference::{InferenceHost, InferenceRoute, run_inference},
};

const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.host.routes.responses";

fn run_responses(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let host = InferenceHost::new(call.view()?, ROUTE_HOST_MODULE);
    run_inference::<ResponsesRoute, _>(py, call, asynchronous, ResponsesPythonHost(host))
}

#[pyfunction]
pub(crate) fn responses(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_responses(py, call, false)
}

#[pyfunction]
pub(crate) fn aresponses(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_responses(py, call, true)
}

impl InferenceRoute for ResponsesRoute {
    type Protocol = Responses;
    const OPERATION: LoggingOperation = LoggingOperation::Responses;
    const SYNC_CALL_TYPE: &'static str = "responses";
    const ASYNC_CALL_TYPE: &'static str = "aresponses";

    fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self::new(http, auth, secrets)
    }

    async fn call(
        self,
        request: ResponsesCall,
        interceptors: ChannelInterceptors<Responses>,
        options: CallOptions,
    ) -> Result<OutputOf<Responses>, RouteError> {
        self.execute(request, &interceptors, options).await
    }
}

impl crate::cache::CachedStream for Responses {
    const TERMINAL_EVENT: Option<&'static str> = Some(<Self as StreamCachable>::TERMINAL_EVENT);

    fn replay(text: String) -> Option<(Self::StreamHead, Vec<Self::Chunk>)> {
        Some((
            litellm_inference_responses::types::ResponsesStreamHead {
                headers: Vec::new(),
            },
            vec![text.into()],
        ))
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        <Self as StreamCachable>::bytes(chunk)
    }
}
