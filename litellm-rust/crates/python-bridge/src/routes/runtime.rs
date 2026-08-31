use std::future::Future;
use std::sync::mpsc::sync_channel;

use litellm_core::error::{CoreError, CoreResult};
use litellm_python_interop::{release_gil, to_py};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;

pub(super) fn run_sync<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(CoreError) -> PyErr,
) -> PyResult<Py<PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = CoreResult<T>> + Send + 'static,
{
    let (sender, receiver) = sync_channel(1);
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let _ = sender.send(future.await);
    });
    let result = release_gil(py, move || receiver.recv())
        .map_err(|_| PyRuntimeError::new_err("native route task terminated"))?
        .map_err(map_error)?;
    to_py(py, &result)
}

pub(super) fn run_async<T, F>(
    py: Python<'_>,
    future: F,
    map_error: fn(CoreError) -> PyErr,
) -> PyResult<Bound<'_, PyAny>>
where
    T: Serialize + Send + 'static,
    F: Future<Output = CoreResult<T>> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = future.await.map_err(map_error)?;
        Python::attach(|py| to_py(py, &result))
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn runtime_error(error: CoreError) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }

    fn extract_bool(py: Python<'_>, result: PyResult<Py<PyAny>>) -> bool {
        result
            .expect("route should complete")
            .bind(py)
            .extract()
            .expect("result should convert")
    }

    #[test]
    fn sync_runner_polls_future_on_tokio_worker() {
        Python::initialize();
        Python::attach(|py| {
            let caller_thread = std::thread::current().id();
            let result = run_sync(
                py,
                async move { Ok(std::thread::current().id() != caller_thread) },
                runtime_error,
            );

            assert!(extract_bool(py, result));
        });
    }

    #[test]
    fn sync_runner_releases_gil_while_waiting() {
        Python::initialize();
        Python::attach(|py| {
            let result = run_sync(
                py,
                async {
                    let gil_acquired = tokio::time::timeout(
                        Duration::from_secs(2),
                        tokio::task::spawn_blocking(|| Python::attach(|_| true)),
                    )
                    .await;
                    Ok(matches!(gil_acquired, Ok(Ok(true))))
                },
                runtime_error,
            );

            assert!(extract_bool(py, result));
        });
    }
}
