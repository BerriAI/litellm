use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub struct PreparedCall {
    callable: Py<PyAny>,
    positional: Py<PyTuple>,
    keywords: Option<Py<PyDict>>,
}

impl PreparedCall {
    pub fn new(callable: Py<PyAny>, positional: Py<PyTuple>, keywords: Option<Py<PyDict>>) -> Self {
        Self {
            callable,
            positional,
            keywords,
        }
    }

    pub fn invoke(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.callable.call(
            py,
            self.positional.bind(py),
            self.keywords.as_ref().map(|kwargs| kwargs.bind(py)),
        )
    }
}
