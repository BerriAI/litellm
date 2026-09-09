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
                operation_binding(operation, asynchronous, Route::Messages).unwrap(),
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
        let complete = operation_binding(
            Operation::Complete(Outcome::Success),
            false,
            Route::Messages,
        )
        .expect_err("complete lifecycle should fail");
        assert_eq!(
            complete.to_string(),
            "RuntimeError: messages lifecycle is complete"
        );
    });
}
