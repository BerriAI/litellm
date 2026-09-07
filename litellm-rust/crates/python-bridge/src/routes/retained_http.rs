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

#[pyfunction]
fn encode(boundary: &Bound<'_, PyAny>, roots: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    invoke(
        boundary,
        BoundaryMethod::Encode.resolve(false),
        PyTuple::new(boundary.py(), [roots])?,
    )
}

fn request(execution: &Bound<'_, PyAny>) -> PyResult<Request> {
    type ByteHeaders<'py> = Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>;
    let encoded = execution.call_method0("encode")?;
    let (url, headers, body, timeout_seconds): (String, ByteHeaders<'_>, Bound<'_, PyBytes>, f64) =
        encoded.extract()?;
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
fn send<'py>(execution: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let request = request(execution)?;
    pyo3_async_runtimes::tokio::future_into_py(execution.py(), async move {
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
    let execution = execution_module(py)?
        .getattr("RetainedExecution")?
        .call1((boundary,))?;
    execution.call_method0("prepare")?;
    let request = request(&execution)?;
    let response = run_sync_value(py, buffered_post::send(request), core_error_to_pyerr)?;
    let wire = Wire(response).into_pyobject(py)?;
    execution.call_method1("finish", (wire,)).map(Bound::unbind)
}

pub(crate) fn run_async(boundary: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let module = execution_module(boundary.py())?;
    let execution = module.getattr("RetainedExecution")?.call1((boundary,))?;
    module
        .getattr("drive")?
        .call1((execution,))
        .map(Bound::unbind)
}

fn execution_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    if MODULE.get(py).is_none() {
        let module = PyModule::from_code(
            py,
            c"class RetainedExecution:
    __slots__ = ('binding', 'roots')

    def __init__(self, binding):
        self.binding = binding
        self.roots = None

    def prepare(self):
        self.roots = _prepare(self.binding, False)

    async def aprepare(self):
        self.roots = await _prepare(self.binding, True)

    def encode(self):
        return _encode(self.binding, self.roots)

    def finish(self, wire):
        return _finish(self.binding, wire, False)

    async def afinish(self, wire):
        return await _finish(self.binding, wire, True)

async def drive(execution):
    await execution.aprepare()
    wire = await _send(execution)
    return await execution.afinish(wire)
",
            c"retained_execution.py",
            c"_retained_execution",
        )?;
        module.add("_prepare", wrap_pyfunction!(prepare, &module)?)?;
        module.add("_encode", wrap_pyfunction!(encode, &module)?)?;
        module.add("_finish", wrap_pyfunction!(finish, &module)?)?;
        module.add("_send", wrap_pyfunction!(send, &module)?)?;
        let _ = MODULE.set(py, module.unbind());
    }
    let module = MODULE.get(py).unwrap();
    Ok(module.bind(py).clone())
}
