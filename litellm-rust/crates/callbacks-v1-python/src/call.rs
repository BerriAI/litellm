use litellm_host::{machine::Machine, route::Route};
use litellm_host_python::{RouteHost, run_call};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::adapter::V1PythonLifecycle;
use crate::registry::snapshot;

#[derive(Clone, Copy, Debug)]
pub struct V1PythonSurface {
    pub call_type: &'static str,
}

pub fn run_v1_python_call<H, M>(
    py: Python<'_>,
    surface: V1PythonSurface,
    kwargs: &Bound<'_, PyDict>,
    machine: M,
    route: H,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = <H::Route as Route>::Response> + 'static,
{
    let subscribers = snapshot(py, asynchronous)?;
    let arguments = kwargs.clone().unbind();
    run_call(
        py,
        machine,
        route,
        Box::new(V1PythonLifecycle::new(surface, subscribers, asynchronous)),
        arguments,
        asynchronous,
    )
}
