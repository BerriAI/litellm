mod host;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

fn run_messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        LoggingOperation::Messages,
        &request,
        &args,
        &kwargs,
        asynchronous,
    )?;
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
                move |(call, selection): (_, Option<crate::cache::PythonCacheConfig>),
                      services,
                      interceptors,
                      observers| async move {
                    let (cache, options) =
                        selection.map(|config| config.into_parts(services)).unzip();
                    route
                        .with_cache(cache)
                        .execute(
                            call,
                            &interceptors,
                            litellm_inference::CallOptions {
                                cache: options,
                                model_group: None,
                                observers,
                            },
                        )
                        .await
                },
            ))
        },
        MessagesPythonHost::new(request.unbind(), asynchronous),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_messages(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn amessages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_messages(py, request, args, kwargs, true)
}
