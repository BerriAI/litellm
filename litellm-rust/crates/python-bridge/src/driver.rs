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
    awaiting: bool,
}

fn operation_binding(
    operation: Operation,
    asynchronous: bool,
    supports_pre_call: bool,
    route: &str,
) -> PyResult<OperationBinding> {
    let binding = match operation {
        Operation::Setup => OperationBinding {
            method: "setup",
            awaiting: false,
        },
        Operation::DeploymentPre => OperationBinding {
            method: "deployment_pre",
            awaiting: true,
        },
        Operation::BuildRequest => OperationBinding {
            method: "build_request",
            awaiting: false,
        },
        Operation::PreCall if supports_pre_call => OperationBinding {
            method: "pre_call",
            awaiting: false,
        },
        Operation::PreCall => {
            return Err(PyRuntimeError::new_err(format!(
                "{route} lifecycle selected an unsupported pre-call operation"
            )));
        }
        Operation::Send if asynchronous => OperationBinding {
            method: "send",
            awaiting: true,
        },
        Operation::Send => OperationBinding {
            method: "send_sync",
            awaiting: false,
        },
        Operation::DeploymentSuccess => OperationBinding {
            method: "deployment_success",
            awaiting: true,
        },
        Operation::DeploymentFailure => OperationBinding {
            method: "deployment_failure",
            awaiting: true,
        },
        Operation::SyncSuccess => OperationBinding {
            method: "sync_success",
            awaiting: false,
        },
        Operation::AsyncSuccess => OperationBinding {
            method: "async_success",
            awaiting: false,
        },
        Operation::SyncSuccessIfNeeded => OperationBinding {
            method: "sync_success_if_needed",
            awaiting: false,
        },
        Operation::SyncFailure => OperationBinding {
            method: "sync_failure",
            awaiting: false,
        },
        Operation::AsyncFailure => OperationBinding {
            method: "async_failure",
            awaiting: true,
        },
        Operation::Restore => OperationBinding {
            method: "restore",
            awaiting: false,
        },
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
    supports_pre_call: bool,
    route: &str,
    host: Py<PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let binding = operation_binding(operation, asynchronous, supports_pre_call, route)?;
    Ok((
        binding.awaiting,
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
                (Operation::Setup, false, false, "setup", false),
                (
                    Operation::DeploymentPre,
                    false,
                    false,
                    "deployment_pre",
                    true,
                ),
                (
                    Operation::BuildRequest,
                    false,
                    false,
                    "build_request",
                    false,
                ),
                (Operation::PreCall, false, true, "pre_call", false),
                (Operation::Send, false, false, "send_sync", false),
                (Operation::Send, true, false, "send", true),
                (
                    Operation::DeploymentSuccess,
                    false,
                    false,
                    "deployment_success",
                    true,
                ),
                (
                    Operation::DeploymentFailure,
                    false,
                    false,
                    "deployment_failure",
                    true,
                ),
                (Operation::SyncSuccess, false, false, "sync_success", false),
                (
                    Operation::AsyncSuccess,
                    false,
                    false,
                    "async_success",
                    false,
                ),
                (
                    Operation::SyncSuccessIfNeeded,
                    false,
                    false,
                    "sync_success_if_needed",
                    false,
                ),
                (Operation::SyncFailure, false, false, "sync_failure", false),
                (Operation::AsyncFailure, false, false, "async_failure", true),
                (Operation::Restore, false, false, "restore", false),
            ];
            for (operation, asynchronous, pre_call, method, awaiting) in cases {
                assert_eq!(
                    operation_binding(operation, asynchronous, pre_call, "test").unwrap(),
                    OperationBinding { method, awaiting },
                );
            }
        });
    }

    #[test]
    fn invalid_operations_raise_route_specific_errors() {
        Python::initialize();
        Python::attach(|_| {
            let pre_call = operation_binding(Operation::PreCall, false, false, "messages")
                .expect_err("unsupported pre-call should fail");
            assert_eq!(
                pre_call.to_string(),
                "RuntimeError: messages lifecycle selected an unsupported pre-call operation"
            );
            let complete = operation_binding(
                Operation::Complete(Outcome::Success),
                false,
                false,
                "messages",
            )
            .expect_err("complete lifecycle should fail");
            assert_eq!(
                complete.to_string(),
                "RuntimeError: messages lifecycle is complete"
            );
        });
    }
}
