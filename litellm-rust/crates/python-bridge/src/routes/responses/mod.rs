mod host;
mod websocket;

use std::sync::Arc;

use host::ResponsesPythonHost;
use litellm_auth::AuthServices;
use litellm_cache_response::ScopedCache;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{call::OutputOf, machine::ChannelInterceptors, protocol::Protocol};
use litellm_inference_responses::{ResponsesRoute, route::Responses};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;
pub(crate) use websocket::ResponsesWebSocketConnection;

use super::{
    NativeCall,
    inference::{InferenceHost, InferenceRoute, run_inference},
};
use crate::cache::Cached;

const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.responses.route_host";

fn run_responses(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let host = InferenceHost::new(
        call.view()?,
        ROUTE_HOST_MODULE,
        "responses",
        if asynchronous {
            "aresponses"
        } else {
            "responses"
        },
        asynchronous,
    );
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
        call: <Responses as Protocol>::Request,
        interceptors: ChannelInterceptors<Cached<Responses>>,
        options: litellm_inference::CallOptions,
    ) -> Result<OutputOf<Responses>, litellm_inference::RouteError> {
        ResponsesRoute::execute(&self, call, &interceptors, options).await
    }
}
