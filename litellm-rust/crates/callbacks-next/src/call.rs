use litellm_host::{machine::Machine, route::Route};
use litellm_host_python::{RouteHost, run_call};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::adapter::NextLifecycle;
use crate::registry::snapshot;

#[derive(Clone, Copy, Debug)]
pub struct NextSurface {
    pub call_type: &'static str,
}

pub fn run_next_call<H, M>(
    py: Python<'_>,
    surface: NextSurface,
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
        Box::new(NextLifecycle::new(surface, subscribers, asynchronous)),
        arguments,
        asynchronous,
    )
}
