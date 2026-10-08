mod host;

use host::ChatCompletionsPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_inference_chat::ChatCompletionsRoute;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use super::codec::RouteCodec;

fn run_chat_completions(
    py: Python<'_>,
    request: Bound<'_, PyDict>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let host = RouteCodec::new(
        request.clone(),
        "litellm.rust_bridge.chat_completions.route_host",
    );
    let cache_call_type = if asynchronous {
        "acompletion"
    } else {
        "completion"
    };
    crate::cache::admit_native(py, &kwargs, cache_call_type)?;
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        LoggingOperation::Completion,
        request.as_any(),
        &args,
        &kwargs,
        asynchronous,
    )?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = ChatCompletionsRoute::new(
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
        ChatCompletionsPythonHost(host),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn completion(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_chat_completions(py, call.bound, call.args, call.kwargs, false)
}

#[pyfunction]
pub(crate) fn acompletion(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_chat_completions(py, call.bound, call.args, call.kwargs, true)
}
