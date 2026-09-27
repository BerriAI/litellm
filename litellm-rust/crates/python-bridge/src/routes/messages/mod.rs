mod host;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::{
    LegacySurface, PassThroughStream, PublicCall, run_legacy_call,
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

const SURFACE: LegacySurface = LegacySurface {
    call_type: "anthropic_messages",
    input_description: "Messages",
    stream: Some(PassThroughStream {
        url_route: "/v1/messages",
        endpoint_type: "anthropic",
    }),
};

fn run_messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let machine = crate::http::call_client(py, &kwargs, asynchronous)?
        .messages_machine()
        .map_err(crate::http::client_error)?;
    run_legacy_call(
        py,
        SURFACE,
        PublicCall::capture(&request, &args, &kwargs)?,
        move |request| crate::logger::LoggedMachine::new(machine(request)),
        MessagesPythonHost::new(request.unbind()),
        crate::preflight::sdk_preflight,
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
