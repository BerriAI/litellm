//! The callback contracts a Python call runs under, composed here once so that no route
//! selects one. Every call runs the legacy `Logging` contract, for as long as Python's
//! `Logging` exists, and then the v1 contract whenever anything is subscribed to it.
//! Both are a [`PythonLifecycle`]; that trait is the whole boundary the driver knows.

use litellm_callbacks_legacy::{LegacyPythonLifecycle, LegacyPythonSurface};
pub(crate) use litellm_callbacks_legacy::{PassThroughStream, PublicCall};
use litellm_callbacks_v1_python::{V1PythonLifecycle, V1PythonSurface};
use litellm_host::{machine::Machine, route::Route};
use litellm_host_python::{LifecycleChain, PythonLifecycle, RouteHost, run_call};
use pyo3::prelude::*;

/// What a route tells the callback contracts about itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CallSurface {
    pub call_type: &'static str,
    /// Legacy only: what `Logging.pre_call` is told the input was.
    pub input_description: &'static str,
    /// Legacy only: how a streamed response is billed; `None` for a route that never
    /// streams.
    pub stream: Option<PassThroughStream>,
    pub updates_logging_before_preparation: bool,
    pub redact_payloads: bool,
}

impl CallSurface {
    fn legacy(self) -> LegacyPythonSurface {
        LegacyPythonSurface {
            call_type: self.call_type,
            input_description: self.input_description,
            stream: self.stream,
            updates_logging_before_preparation: self.updates_logging_before_preparation,
        }
    }

    fn v1(self) -> V1PythonSurface {
        V1PythonSurface {
            call_type: self.call_type,
            redact_payloads: self.redact_payloads,
        }
    }
}

/// Legacy first: v1 subscribers observe the keyword view and the wire request as the
/// legacy contract left them, and share the `litellm_call_id` its setup assigned.
fn lifecycle(
    py: Python<'_>,
    surface: CallSurface,
    call: PublicCall,
    asynchronous: bool,
) -> PyResult<Box<dyn PythonLifecycle>> {
    let legacy = Box::new(LegacyPythonLifecycle::new(
        py,
        surface.legacy(),
        call,
        asynchronous,
    ));
    Ok(
        match V1PythonLifecycle::subscribed(py, surface.v1(), asynchronous)? {
            Some(v1) => Box::new(LifecycleChain::new(vec![legacy, Box::new(v1)])),
            None => legacy,
        },
    )
}

/// Runs one native call for a Python caller. The route host projects from the keyword
/// view the lifecycle's `begin` prepares.
pub(crate) fn run_python_call<H, M>(
    py: Python<'_>,
    surface: CallSurface,
    call: PublicCall,
    machine: M,
    route: H,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: RouteHost + 'static,
    M: Machine<Route = H::Route, Complete = <H::Route as Route>::Response> + 'static,
{
    let arguments = call.arguments(py);
    let lifecycle = lifecycle(py, surface, call, asynchronous)?;
    run_call(py, machine, route, lifecycle, arguments, asynchronous)
}
