use std::future::Future;

use litellm_core::error::Error;
use litellm_python_interop::{release_gil, to_py};
use pyo3::prelude::*;
use serde::Serialize;

use crate::function_trace::trace_call;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)
}

fn block_on<T>(
    py: Python<'_>,
    call: impl Future<Output = Result<T, Error>> + Send,
    trace: bool,
    map_err: fn(Error) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send,
{
    let result = release_gil(py, || {
        pyo3_async_runtimes::tokio::get_runtime().block_on(trace_call(call, trace))
    });
    match result {
        Ok(response) => to_py(py, &response),
        Err(err) => Err(map_err(err)),
    }
}

fn into_py_future<'py, T>(
    py: Python<'py>,
    call: impl Future<Output = Result<T, Error>> + Send + 'static,
    trace: bool,
    map_err: fn(Error) -> PyErr,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Serialize + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let response = trace_call(call, trace).await.map_err(map_err)?;
        Python::attach(|py| to_py(py, &response))
    })
}
