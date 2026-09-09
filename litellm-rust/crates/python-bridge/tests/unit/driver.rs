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

#[test]
fn bridge_drives_real_callback_traces_without_exposing_native_input_hooks() {
    use litellm_core::lifecycle::program::{CallLifecycle, Observations, ProgramOptions};
    Python::initialize();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            c"
class Host:
    def __init__(self, reject):
        self.events = []
        self.reject = reject
    def __getattr__(self, name):
        if name == 'input_hooks':
            raise AssertionError('native input hooks must not invoke Python callbacks')
        def callback():
            self.events.append(name)
            if name == self.reject:
                raise ValueError(name)
        return callback
",
            Some(&globals),
            Some(&globals),
        )
        .unwrap();
        for asynchronous in [false, true] {
            for reject in [
                None,
                Some("build_request"),
                Some("pre_call"),
                Some("send"),
                Some("send_sync"),
            ] {
                let host = globals
                    .get_item("Host")
                    .unwrap()
                    .unwrap()
                    .call1((reject,))
                    .unwrap()
                    .unbind();
                let mut machine = CallLifecycle::planned(ProgramOptions {
                    asynchronous,
                    internal_call: false,
                });
                while !matches!(machine.operation(), Operation::Complete(_)) {
                    let result = invoke(
                        py,
                        machine.operation(),
                        asynchronous,
                        Route::Messages,
                        host.clone_ref(py),
                    );
                    let ticket = machine.issue().unwrap();
                    machine.complete_operation(ticket,
                        if result.is_ok() {
                            Outcome::Success
                        } else {
                            Outcome::Failure
                        },
                        Observations {
                            logger_available: true,
                            has_fallbacks: false,
                        },
                    );
                }
                let events: Vec<String> = host.getattr(py, "events").unwrap().extract(py).unwrap();
                let send = if asynchronous { "send" } else { "send_sync" };
                let mut expected = vec!["setup"];
                if asynchronous {
                    expected.push("deployment_pre");
                }
                expected.extend(["build_request", "pre_call", send]);
                if let Some(position) = expected.iter().position(|event| Some(*event) == reject) {
                    expected.truncate(position + 1);
                    if asynchronous {
                        expected.push("deployment_failure");
                    }
                    expected.push("sync_failure");
                    if asynchronous {
                        expected.push("async_failure");
                    }
                } else if asynchronous {
                    expected.extend([
                        "deployment_success",
                        "async_success",
                        "sync_success_if_needed",
                    ]);
                } else {
                    expected.push("sync_success");
                }
                expected.push("restore");
                assert_eq!(events, expected, "async={asynchronous}, reject={reject:?}");
            }
        }
    });
}
