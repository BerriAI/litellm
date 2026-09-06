use std::cell::Cell;
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use litellm_core::observability::{FunctionTrace, FunctionTraceEvent};
use litellm_python_interop::Pythonized;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Serialize;
use tracing::instrument::WithSubscriber;

thread_local! {
    static ACTIVE_CAPTURE: Cell<Option<u64>> = const { Cell::new(None) };
}

static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(1);
static CAPTURES: OnceLock<Mutex<HashMap<u64, FunctionTrace>>> = OnceLock::new();

#[derive(Serialize)]
pub(crate) struct TracedResponse<T> {
    response: T,
    trace: Vec<FunctionTraceEvent>,
}

pub(crate) async fn capture<T, E>(
    future: impl Future<Output = Result<T, E>>,
) -> Result<TracedResponse<T>, E> {
    let trace = FunctionTrace::default();
    let response = future.with_subscriber(trace.dispatcher()).await?;
    Ok(TracedResponse {
        response,
        trace: trace.events(),
    })
}

pub(crate) fn capture_active<T, E>(
    future: impl Future<Output = Result<T, E>>,
) -> impl Future<Output = Result<T, E>> {
    let trace = ACTIVE_CAPTURE.with(|active| {
        active.get().and_then(|capture_id| {
            captures()
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(&capture_id)
                .cloned()
        })
    });
    async move {
        match trace {
            Some(trace) => future.with_subscriber(trace.dispatcher()).await,
            None => future.await,
        }
    }
}

fn captures() -> &'static Mutex<HashMap<u64, FunctionTrace>> {
    CAPTURES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[pyfunction]
fn start_capture() -> PyResult<u64> {
    ACTIVE_CAPTURE.with(|active| {
        if active.get().is_some() {
            return Err(PyRuntimeError::new_err(
                "a native trace capture is already active",
            ));
        }
        let capture_id = NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed);
        captures()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(capture_id, FunctionTrace::default());
        active.set(Some(capture_id));
        Ok(capture_id)
    })
}

#[pyfunction]
fn finish_capture(py: Python<'_>, capture_id: u64) -> PyResult<Py<PyAny>> {
    ACTIVE_CAPTURE.with(|active| {
        if active.get() != Some(capture_id) {
            return Err(PyRuntimeError::new_err(
                "native trace capture is not active on this thread",
            ));
        }
        active.set(None);
        let trace = captures()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&capture_id)
            .ok_or_else(|| PyRuntimeError::new_err("native trace capture does not exist"))?;
        let events = trace.events();
        if events.is_empty() {
            return Err(PyRuntimeError::new_err(
                "native trace capture did not observe a production bridge route",
            ));
        }
        Pythonized(events).into_pyobject(py).map(Bound::unbind)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_capture, module)?)?;
    module.add_function(wrap_pyfunction!(finish_capture, module)?)?;
    Ok(())
}
