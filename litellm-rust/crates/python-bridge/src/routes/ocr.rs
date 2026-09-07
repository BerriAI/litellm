//! Retained OCR route: Python owns request/response objects, Rust sequences
//! prepare -> encode -> POST -> finish through owning `Py` handles.

use litellm_core::http_utils::buffered_post::{self, Request, Response};
use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyTuple};

use crate::errors::core_error_to_pyerr;
use crate::execution::{run_async_value, run_sync_value};

#[derive(Clone, Copy)]
struct BoundaryStep {
    method: &'static str,
    awaited: bool,
}

const PREPARE_SYNC: BoundaryStep = BoundaryStep {
    method: "prepare",
    awaited: false,
};
const PREPARE_ASYNC: BoundaryStep = BoundaryStep {
    method: "aprepare",
    awaited: true,
};
const ENCODE: BoundaryStep = BoundaryStep {
    method: "encode",
    awaited: false,
};
const FINISH_SYNC: BoundaryStep = BoundaryStep {
    method: "finish",
    awaited: false,
};
const FINISH_ASYNC: BoundaryStep = BoundaryStep {
    method: "afinish",
    awaited: true,
};

fn invoke(
    boundary: &Bound<'_, PyAny>,
    step: BoundaryStep,
    args: Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let call = PreparedCall::new(
        if step.awaited {
            InvocationMode::Await
        } else {
            InvocationMode::Direct
        },
        boundary.getattr(step.method)?.unbind(),
        args.unbind(),
        None,
    );
    match call.invoke(boundary.py())? {
        InvocationOutcome::Returned(value) | InvocationOutcome::Awaitable(value) => Ok(value),
    }
}

#[pyfunction]
fn prepare(boundary: &Bound<'_, PyAny>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let step = if asynchronous {
        PREPARE_ASYNC
    } else {
        PREPARE_SYNC
    };
    invoke(boundary, step, PyTuple::empty(boundary.py()))
}

fn request(boundary: &Bound<'_, PyAny>, roots: &Bound<'_, PyAny>) -> PyResult<Request> {
    type ByteHeaders<'py> = Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>;
    let py = boundary.py();
    let encoded = invoke(boundary, ENCODE, PyTuple::new(py, [roots])?)?;
    let (url, headers, body, timeout_seconds): (String, ByteHeaders<'_>, Bound<'_, PyBytes>, f64) =
        encoded.into_bound(py).extract()?;
    Ok(Request {
        url,
        headers: headers
            .into_iter()
            .map(|(name, value)| (name.as_bytes().to_vec(), value.as_bytes().to_vec()))
            .collect(),
        body: body.as_bytes().to_vec(),
        timeout_seconds,
    })
}

struct Wire(Response);

impl<'py> IntoPyObject<'py> for Wire {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let headers = PyList::new(
            py,
            self.0
                .headers
                .iter()
                .map(|(name, value)| (PyBytes::new(py, name), PyBytes::new(py, value))),
        )?;
        (self.0.status, headers, PyBytes::new(py, &self.0.content)).into_pyobject(py)
    }
}

#[pyfunction]
fn send<'a>(
    boundary: &'a Bound<'a, PyAny>,
    roots: &Bound<'_, PyAny>,
) -> PyResult<Bound<'a, PyAny>> {
    let request = request(boundary, roots)?;
    pyo3_async_runtimes::tokio::future_into_py(boundary.py(), async move {
        let response = run_async_value(buffered_post::send(request), core_error_to_pyerr).await?;
        Ok(Wire(response))
    })
}

#[pyfunction]
fn finish(
    boundary: &Bound<'_, PyAny>,
    wire: &Bound<'_, PyAny>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let step = if asynchronous {
        FINISH_ASYNC
    } else {
        FINISH_SYNC
    };
    invoke(boundary, step, PyTuple::new(boundary.py(), [wire])?)
}

#[pyfunction]
fn ocr(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = boundary.py();
    let roots = prepare(boundary, false)?;
    let request = request(boundary, roots.bind(py))?;
    let response = run_sync_value(py, buffered_post::send(request), core_error_to_pyerr)?;
    let wire = Wire(response).into_pyobject(py)?;
    finish(boundary, &wire, false)
}

#[pyfunction]
fn aocr<'a>(boundary: &'a Bound<'a, PyAny>) -> PyResult<Bound<'a, PyAny>> {
    driver(boundary.py())?.getattr("drive")?.call1((boundary,))
}

/// The async route must await `aprepare`/`afinish` inline in the caller's
/// Python task, so a Python driver coroutine owns the roots between steps.
/// Compiling the driver runs Python (audit hooks can re-enter `aocr`), so
/// compile first and publish only a finished module into the once-lock.
fn driver(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static DRIVER: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if let Some(module) = DRIVER.get(py) {
        return Ok(module.bind(py));
    }
    let module = PyModule::from_code(
        py,
        c"async def drive(boundary):
    roots = await _prepare(boundary, True)
    wire = await _send(boundary, roots)
    return await _finish(boundary, wire, True)
",
        c"ocr_driver.py",
        c"_ocr_driver",
    )?;
    module.add("_prepare", wrap_pyfunction!(prepare, &module)?)?;
    module.add("_send", wrap_pyfunction!(send, &module)?)?;
    module.add("_finish", wrap_pyfunction!(finish, &module)?)?;
    Ok(DRIVER.get_or_init(py, || module.unbind()).bind(py))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!(ocr, module)?)?;
    crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!(aocr, module)?)?;
    Ok(())
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}
