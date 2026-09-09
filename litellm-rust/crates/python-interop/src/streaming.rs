use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::{Stream, StreamExt};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::sync::{Mutex, Notify};

pub type PythonByteStream = Pin<Box<dyn Stream<Item = PyResult<Vec<u8>>> + Send>>;
pub type PythonStreamCompletion = Pin<Box<dyn Future<Output = PyResult<()>> + Send>>;

struct State {
    stream: Mutex<Option<PythonByteStream>>,
    completion: Mutex<Option<PythonStreamCompletion>>,
    closed: AtomicBool,
    polling: AtomicBool,
    notify: Notify,
}

impl State {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
        if let Ok(mut stream) = self.stream.try_lock() {
            stream.take();
        }
    }

    async fn finish(&self) -> PyResult<()> {
        let mut completion = self.completion.lock().await;
        if let Some(future) = completion.as_mut() {
            let result = future.await;
            completion.take();
            result?;
        }
        Ok(())
    }
}

struct PollGuard {
    state: Arc<State>,
    completed: bool,
}

impl Drop for PollGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.state.close();
        }
        self.state.polling.store(false, Ordering::Release);
    }
}

#[pyclass]
pub struct AsyncByteStream {
    state: Arc<State>,
}

impl AsyncByteStream {
    pub fn new(stream: PythonByteStream, completion: PythonStreamCompletion) -> Self {
        Self {
            state: Arc::new(State {
                stream: Mutex::new(Some(stream)),
                completion: Mutex::new(Some(completion)),
                closed: AtomicBool::new(false),
                polling: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }
}

impl Drop for AsyncByteStream {
    fn drop(&mut self) {
        self.state.close();
    }
}

#[pymethods]
impl AsyncByteStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let state = self.state.clone();
        let future = crate::run_async_py(py, async move {
            if state.polling.swap(true, Ordering::AcqRel) {
                return Err(PyRuntimeError::new_err(
                    "concurrent stream iteration is not supported",
                ));
            }
            let mut guard = PollGuard {
                state: state.clone(),
                completed: false,
            };
            let item = {
                let mut stream = state.stream.lock().await;
                if state.closed.load(Ordering::Acquire) {
                    stream.take();
                }
                match stream.as_mut() {
                    Some(inner) => {
                        tokio::select! {
                            biased;
                            _ = state.notify.notified() => None,
                            item = inner.next() => item,
                        }
                    }
                    None => None,
                }
            };
            match item {
                Some(Ok(bytes)) => {
                    guard.completed = true;
                    Python::attach(|py| Ok(PyBytes::new(py, &bytes).unbind()))
                }
                item => {
                    state.close();
                    state.finish().await?;
                    guard.completed = true;
                    Err(match item {
                        Some(Err(error)) => error,
                        _ => PyStopAsyncIteration::new_err(()),
                    })
                }
            }
        })?;
        Ok(future.unbind())
    }

    fn aclose(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.state.close();
        let state = self.state.clone();
        Ok(crate::run_async_py(py, async move {
            state.stream.lock().await.take();
            state.finish().await
        })?
        .unbind())
    }
}
