mod host;

use host::MessagesRouteHost;
use litellm_core::messages::route::{messages_machine, supports};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{
    callbacks::{CallSurface, PassThroughStream, PublicCall, run_python_call},
    errors::RustBridgeDeclined,
};

const SURFACE: CallSurface = CallSurface {
    call_type: "anthropic_messages",
    input_description: "Messages",
    stream: Some(PassThroughStream {
        url_route: "/v1/messages",
        endpoint_type: "anthropic",
    }),
    updates_logging_before_preparation: false,
    redact_payloads: false,
};

fn run_messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let model: String = request.getattr("model")?.extract()?;
    let provider: Option<String> = request.getattr("custom_llm_provider")?.extract()?;
    let stream = request
        .getattr("stream")?
        .extract::<Option<bool>>()?
        .unwrap_or(false);
    if !supports(&model, provider.as_deref(), stream) {
        return Err(RustBridgeDeclined::new_err(
            "the Rust Messages route does not serve this provider",
        ));
    }
    run_python_call(
        py,
        SURFACE,
        PublicCall::capture(&request, &args, &kwargs)?,
        messages_machine(),
        MessagesRouteHost::new(request.unbind()),
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
