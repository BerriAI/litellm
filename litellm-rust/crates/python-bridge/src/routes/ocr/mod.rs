mod document;
mod errors;
mod host;
mod project;

use litellm_callbacks_legacy::{LegacySurface, PublicCall, run_legacy_call};
use litellm_core::ocr::{OcrClient, ocr_machine};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use host::OcrRouteHost;

const SURFACE: LegacySurface = LegacySurface {
    call_type: "ocr",
    input_description: "OCR document processing",
};

const ASYNC_SURFACE: LegacySurface = LegacySurface {
    call_type: "aocr",
    ..SURFACE
};

fn run_ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let client = OcrClient::shared().map_err(errors::to_pyerr)?;
    run_legacy_call(
        py,
        if asynchronous { ASYNC_SURFACE } else { SURFACE },
        PublicCall::capture(&request, &args, &kwargs)?,
        ocr_machine(client),
        OcrRouteHost::new(request.unbind()),
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn aocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, true)
}
