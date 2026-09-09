use litellm_core::lifecycle::program::Operation;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{Error, Route};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OperationBinding {
    method: &'static str,
}

fn operation_binding(
    operation: Operation,
    asynchronous: bool,
    route: Route,
) -> PyResult<OperationBinding> {
    let binding = match operation {
        Operation::Setup => OperationBinding { method: "setup" },
        Operation::DeploymentPre => OperationBinding {
            method: "deployment_pre",
        },
        Operation::BuildRequest => OperationBinding {
            method: "build_request",
        },
        Operation::PreCall => OperationBinding { method: "pre_call" },
        Operation::Send if asynchronous => OperationBinding { method: "send" },
        Operation::Send => OperationBinding {
            method: "send_sync",
        },
        Operation::DeploymentSuccess => OperationBinding {
            method: "deployment_success",
        },
        Operation::DeploymentFailure => OperationBinding {
            method: "deployment_failure",
        },
        Operation::SyncSuccess => OperationBinding {
            method: "sync_success",
        },
        Operation::AsyncSuccess => OperationBinding {
            method: "async_success",
        },
        Operation::SyncSuccessIfNeeded => OperationBinding {
            method: "sync_success_if_needed",
        },
        Operation::SyncFailure => OperationBinding {
            method: "sync_failure",
        },
        Operation::AsyncFailure => OperationBinding {
            method: "async_failure",
        },
        Operation::Restore => OperationBinding { method: "restore" },
        Operation::Complete(_) => {
            return Err(Error::LifecycleComplete(route).into());
        }
    };
    Ok(binding)
}

pub(crate) fn invoke(
    py: Python<'_>,
    operation: Operation,
    asynchronous: bool,
    route: Route,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let binding = operation_binding(operation, asynchronous, route)?;
    Ok((
        operation.is_awaited(asynchronous),
        host.getattr(py, binding.method)?.call0(py)?,
    ))
}

pub(crate) fn drive_sync<'py>(
    py: Python<'py>,
    runner: &Bound<'py, PyModule>,
    arguments: Py<PyDict>,
    bindings: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    runner.call_method1(pyo3::intern!(py, "_drive_sync"), (arguments, bindings))
}

pub(crate) fn drive_async<'py>(
    py: Python<'py>,
    runner: &Bound<'py, PyModule>,
    arguments: Py<PyDict>,
    bindings: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyAny>> {
    runner.call_method1(pyo3::intern!(py, "_drive_async"), (arguments, bindings))
}

#[cfg(test)]
#[path = "../tests/unit/driver.rs"]
mod tests;
