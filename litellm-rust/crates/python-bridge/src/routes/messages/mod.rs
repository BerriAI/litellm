mod host;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use pyo3::prelude::*;

use super::NativeCall;

fn run_messages(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) =
        crate::routes::call_hooks(py, LoggingOperation::Messages, &call, asynchronous)?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = litellm_inference_messages::MessagesRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            Ok(litellm_host::call::hosted_call(
                request,
                None,
                move |(call, selection): (_, crate::cache::Selection),
                      services,
                      interceptors,
                      observers| async move {
                    let (cache, options) = selection.attach(services);
                    let route = match cache {
                        Some(cache) => route.with_cache(cache),
                        None => route,
                    };
                    route
                        .execute(
                            call,
                            &interceptors,
                            litellm_inference::CallOptions {
                                cache: Some(options.policy),
                                observers,
                            },
                        )
                        .await
                },
            ))
        },
        MessagesPythonHost::new(call.view()?, asynchronous),
        hooks,
        asynchronous,
    )
}

/// Python's sync Messages handler raises for every native provider before it sends anything
/// or opens a logging object (`BaseLLMHTTPHandler.anthropic_messages_handler`), and Rust copies
/// that until both sides serve sync calls.
const SYNC_NOT_IMPLEMENTED: &str = "anthropic_messages_handler is not implemented for sync calls";

#[pyfunction]
#[pyo3(signature = (call))]
pub(crate) fn messages(call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    drop(call);
    Err(pyo3::exceptions::PyValueError::new_err(
        SYNC_NOT_IMPLEMENTED,
    ))
}

#[pyfunction]
pub(crate) fn amessages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, true)
}
