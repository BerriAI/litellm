mod host;
mod project;

use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_inference_tokenize::TokenizeRoute;
use pyo3::prelude::*;

use super::NativeCall;
use host::TokenizePythonHost;

fn run_tokenize(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) =
        super::call_hooks(py, LoggingOperation::Tokenize, &call, asynchronous)?;
    super::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let client = crate::http::provider_client(py, arguments, asynchronous)?
                .map_err(crate::http::client_error)?;
            let route = TokenizeRoute::new(
                client,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            Ok(route.machine(request, None))
        },
        TokenizePythonHost::new(call.view()?),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn tokenize(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_tokenize(py, call, false)
}

#[pyfunction]
pub(crate) fn atokenize(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_tokenize(py, call, true)
}
