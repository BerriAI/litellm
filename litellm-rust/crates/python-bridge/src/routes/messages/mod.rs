mod host;

use std::sync::Arc;

use host::MessagesPythonHost;
use litellm_auth::AuthServices;
use litellm_cache_response::ScopedCache;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{call::OutputOf, machine::ChannelInterceptors};
use litellm_inference::{CallOptions, RouteError};
use litellm_inference_messages::{MessagesCall, MessagesRoute, route::Messages};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;

use super::{
    NativeCall,
    inference::{InferenceRoute, run_inference},
};
use crate::cache::Cached;

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

    fn with_cache(self, cache: ScopedCache) -> Self {
        self.with_cache(cache)
    }

    async fn call(
        self,
        request: MessagesCall,
        interceptors: ChannelInterceptors<Cached<Messages>>,
        options: CallOptions,
    ) -> Result<OutputOf<Messages>, RouteError> {
        self.execute(request, &interceptors, options).await
    }
}
