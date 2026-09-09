use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyDict;

use crate::errors::{Error, Route, StateSlot};

pub(crate) struct RetainedCallback {
    roots: Option<RequestRoots>,
    logging: Option<Py<PyAny>>,
    pre_call: Option<Py<PyDict>>,
}

impl RetainedCallback {
    #[cfg(test)]
    pub(crate) fn cleared() -> Self {
        Self {
            roots: None,
            logging: None,
            pre_call: None,
        }
    }

    pub(crate) fn new(roots: RequestRoots, logging: Py<PyAny>, pre_call: Py<PyDict>) -> Self {
        Self {
            roots: Some(roots),
            logging: Some(logging),
            pre_call: Some(pre_call),
        }
    }

    pub(crate) fn roots(&self, route: Route) -> PyResult<&RequestRoots> {
        self.roots
            .as_ref()
            .ok_or_else(|| Error::StateCleared(route, StateSlot::Roots).into())
    }

    pub(crate) fn logging(&self, py: Python<'_>, route: Route) -> PyResult<Py<PyAny>> {
        self.logging
            .as_ref()
            .ok_or_else(|| Error::StateCleared(route, StateSlot::Logging).into())
            .map(|logging| logging.clone_ref(py))
    }

    pub(crate) fn pre_call(&self, py: Python<'_>, route: Route) -> PyResult<Py<PyDict>> {
        self.pre_call
            .as_ref()
            .ok_or_else(|| Error::StateCleared(route, StateSlot::PreCall).into())
            .map(|arguments| arguments.clone_ref(py))
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(roots) = &self.roots {
            roots.traverse(visit)?;
        }
        visit.call(&self.logging)?;
        visit.call(&self.pre_call)
    }

    pub(crate) fn clear(
        &mut self,
    ) -> (Option<RequestRoots>, Option<Py<PyAny>>, Option<Py<PyDict>>) {
        (self.roots.take(), self.logging.take(), self.pre_call.take())
    }
}

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
