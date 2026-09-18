mod document;
mod errors;
mod host;
mod project;

use litellm_callbacks_legacy::{LegacyLogging, LegacySurface};
use litellm_core::ocr::{OcrClient, ocr_machine};
use litellm_host_python::run_call;
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
    let arguments = kwargs.copy()?.unbind();
    let adapter = LegacyLogging::new(
        py,
        if asynchronous { ASYNC_SURFACE } else { SURFACE },
        args.unbind(),
        arguments.clone_ref(py),
        request.clone().unbind(),
        asynchronous,
    );
    run_call(
        py,
        ocr_machine(client),
        OcrRouteHost::new(request.unbind()),
        Box::new(adapter),
        arguments,
        asynchronous,
    )
}

#[pyfunction]
fn ocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, false)
}

#[pyfunction]
fn aocr(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_ocr(py, request, args, kwargs, true)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ocr, module)?)?;
    module.add_function(wrap_pyfunction!(aocr, module)?)
}
