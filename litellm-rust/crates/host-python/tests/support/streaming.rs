use std::convert::Infallible;

use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, hosted_call},
    machine::MachineFault,
    protocol::Protocol,
};
use litellm_host_python::{
    CallOptions, HookChain, InvokeError, PythonBinding, PythonHostCalls, PythonOwned, run_call,
};
use pyo3::{
    exceptions::PyRuntimeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyModule},
};

struct Streaming;

impl Protocol for Streaming {
    type Request = usize;
    type Response = ();
    type Error = StreamingError;
    type HostCall = Infallible;
    type Chunk = usize;
    type StreamHead = ();
}

#[derive(Clone, Debug)]
struct StreamingError;

impl std::fmt::Display for StreamingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("synthetic streaming error")
    }
}

impl From<MachineFault> for StreamingError {
    fn from(_: MachineFault) -> Self {
        Self
    }
}

struct StreamingBinding;

impl PythonBinding for StreamingBinding {
    type Protocol = Streaming;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        _: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<usize, InvokeError<StreamingError>> {
        Ok(arguments
            .get_item("chunks")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("chunks"))?
            .extract()?)
    }

    fn encode_response(&mut self, py: Python<'_>, _: ()) -> PyResult<Py<PyAny>> {
        Ok(py.None())
    }

    fn encode_stream_head(&mut self, py: Python<'_>, _: ()) -> PyResult<Py<PyAny>> {
        Ok(py.None())
    }

    fn encode_chunk(&mut self, py: Python<'_>, chunk: usize) -> PyResult<Py<PyAny>> {
        Ok(chunk.into_pyobject(py)?.into_any().unbind())
    }

    fn map_error(&self, _: Python<'_>, error: StreamingError) -> PyResult<PyErr> {
        Ok(PyRuntimeError::new_err(error.to_string()))
    }

    fn host_error(_: &PyErr) -> StreamingError {
        StreamingError
    }
}

impl PythonHostCalls<Streaming> for StreamingBinding {
    fn handle_host_call(
        &mut self,
        _: Python<'_>,
        operation: Infallible,
    ) -> Result<(), InvokeError<StreamingError>> {
        match operation {}
    }
}

impl PythonOwned for StreamingBinding {
    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

#[pyfunction]
fn native_stream(py: Python<'_>, chunks: usize) -> PyResult<Py<PyAny>> {
    let arguments = PyDict::new(py);
    arguments.set_item("chunks", chunks)?;
    run_call(
        py,
        |_, _, chunks| {
            Ok(hosted_call(chunks, None, |chunks, _, _, _| async move {
                Ok(CallOutput::Stream {
                    head: (),
                    chunks: Box::pin(stream::iter(0..chunks).then(|chunk| async move {
                        tokio::task::yield_now().await;
                        Ok(chunk)
                    })),
                })
            }))
        },
        StreamingBinding,
        HookChain::new(),
        arguments.unbind(),
        CallOptions::new(true, lifecycle_binding),
    )
}

pub fn install(py: Python<'_>, namespace: &Bound<'_, PyDict>) -> PyResult<()> {
    py.import("sys")?
        .getattr("modules")?
        .set_item(LIFECYCLE_MODULE, compile_lifecycle(py)?)?;
    let module = PyModule::new(py, "streaming_test")?;
    module.add_function(pyo3::wrap_pyfunction!(native_stream, &module)?)?;
    namespace.set_item("native_stream", module.getattr("native_stream")?)?;
    Ok(())
}

const LIFECYCLE_MODULE: &str = "streaming_test_lifecycle";

fn lifecycle_binding(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import(LIFECYCLE_MODULE)
}

fn compile_lifecycle(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let source = std::ffi::CString::new(include_str!(
        "../../../../../litellm/rust_bridge/lifecycle.py"
    ))
    .expect("lifecycle source has no null bytes");
    PyModule::from_code(py, &source, c"lifecycle.py", c"streaming_test_lifecycle")
}
