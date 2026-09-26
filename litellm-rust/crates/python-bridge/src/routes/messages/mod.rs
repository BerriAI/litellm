mod host;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::{
    LegacySurface, PassThroughStream, PublicCall, run_legacy_call,
};
use litellm_core::messages::route::messages_machine;
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
    let secrets = crate::secrets::source(py)?;
    run_legacy_call(
        py,
        SURFACE,
        PublicCall::capture(&request, &args, &kwargs)?,
        crate::logger::LoggedMachine::new(messages_machine(secrets)),
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
