mod host;
mod websocket;

use std::sync::Arc;

use host::ResponsesPythonHost;
use litellm_auth::AuthServices;
use litellm_cache_response::{CachePolicy, ScopedCache};
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_host::{call::HostedMachine, protocol::Protocol};
use litellm_host_python::present;
use litellm_inference_responses::{ResponsesRoute, route::Responses};
use litellm_secrets::source::SecretSource;
use pyo3::prelude::*;
use serde_json::Value;
pub(crate) use websocket::ResponsesWebSocketConnection;

use super::{
    NativeCall,
    inference::{InferenceHost, InferenceRoute, run_inference},
};
use crate::errors::RustBridgeDeclined;

const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.responses.route_host";

fn run_responses(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let resolved = call.resolved()?;
    if let Some(reason) = py
        .import(ROUTE_HOST_MODULE)?
        .getattr("decline_reason")?
        .call1((&resolved,))?
        .extract::<Option<String>>()?
    {
        return Err(RustBridgeDeclined::new_err(reason));
    }
    let argument = |name: &str| present(&call.kwargs, &call.base, name);
    let model = argument("model")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("model is required"))?
        .extract::<String>()?;
    let provider = argument("custom_llm_provider")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    if provider
        .as_deref()
        .is_some_and(|provider| provider != "openai")
        || model
            .strip_prefix("openai/")
            .unwrap_or(&model)
            .contains('/')
    {
        return Err(RustBridgeDeclined::new_err(
            "native HTTP responses provider",
        ));
    }
    if argument("stream")?
        .map(|value| litellm_host_python::from_py::<Value>(&value))
        .transpose()?
        .is_some_and(|value| value == Value::Bool(true))
    {
        return Err(RustBridgeDeclined::new_err(
            "native Python responses streaming",
        ));
    }
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

    fn with_cache(self, cache: ScopedCache) -> Self {
        self.with_cache(cache)
    }

    fn machine(
        self,
        call: <Responses as Protocol>::Request,
        policy: CachePolicy,
    ) -> HostedMachine<Responses> {
        self.machine(call, policy)
    }
}
