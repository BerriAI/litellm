mod host;
mod websocket;

use host::ResponsesPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use serde_json::Value;
pub(crate) use websocket::ResponsesWebSocketConnection;

use super::inference::InferenceHost;
use crate::errors::RustBridgeDeclined;

fn run_responses(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let host = InferenceHost::new(
        request.clone(),
        "litellm.rust_bridge.responses.route_host",
        &kwargs,
    )?;
    if let Some(reason) = py
        .import("litellm.rust_bridge.responses.route_host")?
        .getattr("decline_reason")?
        .call1((&request,))?
        .extract::<Option<String>>()?
    {
        return Err(RustBridgeDeclined::new_err(reason));
    }
    let model = host
        .argument(py, &kwargs, "model")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("model is required"))?
        .extract::<String>()?;
    let provider = host
        .argument(py, &kwargs, "custom_llm_provider")?
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
    if host
        .argument(py, &kwargs, "stream")?
        .map(|value| litellm_host_python::from_py::<Value>(&value))
        .transpose()?
        .is_some_and(|value| value == Value::Bool(true))
    {
        return Err(RustBridgeDeclined::new_err(
            "native Python responses streaming",
        ));
    }
    let cache_call_type = if asynchronous {
        "aresponses"
    } else {
        "responses"
    };
    crate::cache::admit_native(py, &kwargs, cache_call_type)?;
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        LoggingOperation::Responses,
        request.as_any(),
        &args,
        &kwargs,
        asynchronous,
    )?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = litellm_inference_responses::ResponsesRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            let (cache, cache_options) =
                crate::cache::configured_native(py, arguments, cache_call_type)?;
            let route = match cache {
                Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
                    cache,
                    litellm_cache_response::CacheScope::Shared,
                )),
                None => route,
            };
            Ok(route.machine(request, cache_options.policy))
        },
        ResponsesPythonHost(host),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn responses(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_responses(py, call.bound, call.args, call.kwargs, false)
}

#[pyfunction]
pub(crate) fn aresponses(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_responses(py, call.bound, call.args, call.kwargs, true)
}
