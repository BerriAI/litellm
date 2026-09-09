use litellm_core::lifecycle::program::Operation;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) const ADDITIONAL_ARGS: &str = "additional_args";
pub(crate) const API_BASE: &str = "api_base";
pub(crate) const API_KEY: &str = "api_key";
pub(crate) const COMPLETE_INPUT_DICT: &str = "complete_input_dict";
pub(crate) const HEADERS: &str = "headers";
pub(crate) const INPUT: &str = "input";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OperationBinding {
    method: &'static str,
}

fn operation_binding(
    operation: Operation,
    asynchronous: bool,
    route: &str,
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
            return Err(PyRuntimeError::new_err(format!(
                "{route} lifecycle is complete"
            )));
        }
    };
    Ok(binding)
}

pub(crate) fn invoke(
    py: Python<'_>,
    operation: Operation,
    asynchronous: bool,
    route: &str,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let binding = operation_binding(operation, asynchronous, route)?;
    Ok((
        operation.is_awaited(asynchronous),
        host.getattr(py, binding.method)?.call0(py)?,
    ))
}

#[cfg(test)]
mod tests {
    use litellm_core::lifecycle::Outcome;

    use super::*;

    #[test]
    fn operation_bindings_cover_the_lifecycle_contract() {
        Python::initialize();
        Python::attach(|_| {
            let cases = [
                (Operation::Setup, false, "setup", false),
                (Operation::DeploymentPre, false, "deployment_pre", true),
                (Operation::BuildRequest, false, "build_request", false),
                (Operation::PreCall, false, "pre_call", false),
                (Operation::Send, false, "send_sync", false),
                (Operation::Send, true, "send", true),
                (
                    Operation::DeploymentSuccess,
                    false,
                    "deployment_success",
                    true,
                ),
                (
                    Operation::DeploymentFailure,
                    false,
                    "deployment_failure",
                    true,
                ),
                (Operation::SyncSuccess, false, "sync_success", false),
                (Operation::AsyncSuccess, false, "async_success", false),
                (
                    Operation::SyncSuccessIfNeeded,
                    false,
                    "sync_success_if_needed",
                    false,
                ),
                (Operation::SyncFailure, false, "sync_failure", false),
                (Operation::AsyncFailure, false, "async_failure", true),
                (Operation::Restore, false, "restore", false),
            ];
            for (operation, asynchronous, method, awaiting) in cases {
                assert_eq!(
                    operation_binding(operation, asynchronous, "test").unwrap(),
                    OperationBinding { method }
                );
                assert_eq!(operation.is_awaited(asynchronous), awaiting);
            }
        });
    }

    #[test]
    fn invalid_operations_raise_route_specific_errors() {
        Python::initialize();
        Python::attach(|_| {
            let complete =
                operation_binding(Operation::Complete(Outcome::Success), false, "messages")
                    .expect_err("complete lifecycle should fail");
            assert_eq!(
                complete.to_string(),
                "RuntimeError: messages lifecycle is complete"
            );
        });
    }
}
