use litellm_core::http_utils::buffered_post::{self, Request, Response};
use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyList, PyTuple};

use crate::errors::core_error_to_pyerr;
use crate::execution::{run_async_value, run_sync_value};

pub(crate) struct MethodBinding {
    pub(crate) name: &'static str,
    pub(crate) mode: InvocationMode,
}

pub(crate) enum BoundaryMethod {
    Prepare,
    Encode,
    Finish,
}

impl BoundaryMethod {
    pub(crate) fn resolve(self, asynchronous: bool) -> MethodBinding {
        match (self, asynchronous) {
            (Self::Prepare, true) => MethodBinding {
                name: "aprepare",
                mode: InvocationMode::Await,
            },
            (Self::Prepare, false) => MethodBinding {
                name: "prepare",
                mode: InvocationMode::Direct,
            },
            (Self::Encode, _) => MethodBinding {
                name: "encode",
                mode: InvocationMode::Direct,
            },
            (Self::Finish, true) => MethodBinding {
                name: "afinish",
                mode: InvocationMode::Await,
            },
            (Self::Finish, false) => MethodBinding {
                name: "finish",
                mode: InvocationMode::Direct,
            },
        }
    }
}

fn invoke(
    boundary: &Bound<'_, PyAny>,
    binding: MethodBinding,
    args: Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let call = PreparedCall::new(
        binding.mode,
        boundary.getattr(binding.name)?.unbind(),
        args.unbind(),
        None,
    );
    match call.invoke(boundary.py())? {
        InvocationOutcome::Returned(value) | InvocationOutcome::Awaitable(value) => Ok(value),
    }
}

#[pyfunction]
fn prepare(boundary: &Bound<'_, PyAny>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    invoke(
        boundary,
        BoundaryMethod::Prepare.resolve(asynchronous),
        PyTuple::empty(boundary.py()),
    )
}

fn encode(boundary: &Bound<'_, PyAny>, roots: &Bound<'_, PyAny>) -> PyResult<Request> {
    type ByteHeaders<'py> = Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>;
    let encoded = invoke(
        boundary,
        BoundaryMethod::Encode.resolve(false),
        PyTuple::new(boundary.py(), [roots])?,
    )?;
    let (url, headers, body, timeout_seconds): (String, ByteHeaders<'_>, Bound<'_, PyBytes>, f64) =
        encoded.extract(boundary.py())?;
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
fn send<'py>(
    boundary: &Bound<'py, PyAny>,
    roots: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let request = encode(boundary, roots)?;
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
    invoke(
        boundary,
        BoundaryMethod::Finish.resolve(asynchronous),
        PyTuple::new(boundary.py(), [wire])?,
    )
}

pub(crate) fn run_sync(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = boundary.py();
    let roots = prepare(boundary, false)?;
    let request = encode(boundary, roots.bind(py))?;
    let response = run_sync_value(py, buffered_post::send(request), core_error_to_pyerr)?;
    let wire = Wire(response).into_pyobject(py)?;
    finish(boundary, wire.as_any(), false)
}

pub(crate) fn run_async(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    static DRIVER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = boundary.py();
    let driver = DRIVER.get_or_try_init(py, || {
        PyModule::from_code(
            py,
            c"async def drive(boundary, prepare, send, finish):
    roots = await prepare(boundary, True)
    wire = await send(boundary, roots)
    return await finish(boundary, wire, True)
",
            c"retained_http_driver.py",
            c"_retained_http_driver",
        )?
        .getattr("drive")
        .map(Bound::unbind)
    })?;
    driver.call1(
        py,
        (
            boundary,
            wrap_pyfunction!(prepare, py)?,
            wrap_pyfunction!(send, py)?,
            wrap_pyfunction!(finish, py)?,
        ),
    )
}
