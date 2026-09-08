use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyDict;

pub(crate) struct RequestRoots {
    arguments: Py<PyDict>,
    body: Py<PyAny>,
    headers: Py<PyAny>,
}

impl RequestRoots {
    pub(crate) fn new(arguments: Py<PyDict>, body: Py<PyAny>, headers: Py<PyAny>) -> Self {
        Self {
            arguments,
            body,
            headers,
        }
    }

    pub(crate) fn arguments<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.arguments.clone_ref(py).into_bound(py)
    }

    pub(crate) fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.body.clone_ref(py).into_bound(py)
    }

    pub(crate) fn headers<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.headers.clone_ref(py).into_bound(py)
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.arguments)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}
