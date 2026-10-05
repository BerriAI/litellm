use pyo3::prelude::*;

/// Why a custom operation the host answered did not produce a result: the route's own code
/// rejected it, which the route classifies like any other native failure, or Python code
/// raised, which reaches the caller as it was raised.
#[derive(Debug)]
pub enum InvokeError<E> {
    Native(E),
    Python(PyErr),
}

impl<E> From<PyErr> for InvokeError<E> {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

pub fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}
