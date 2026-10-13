mod support;

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use litellm_host::{
    hooks::CallHooks,
    lifecycle::{CallEvent, ExecutionEvent, FailureOrigin},
    machine::CallMachine,
    observation::observation_channel,
};
use litellm_host_python::{CallOptions, HookChain, PythonOwned, PythonRuntime, run_call};
use pyo3::{
    exceptions::{
        PyBaseException, PyException, PyLookupError, PyRuntimeError, PyStopIteration, PyTypeError,
        PyValueError, asyncio::CancelledError,
    },
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use rstest::rstest;
use support::{
    HookScript, Interpreter, Log, MachineFactory, OpScript, Synthetic, binding, call_options,
    failing_machine, hooks, interpreter, lifecycle_binding, open_stream, read_all, run_composed,
    run_scripted, settled, success_machine,
};

fn observed(capacity: usize) -> (CallOptions, impl FnMut() -> Vec<CallEvent>) {
    let (sender, mut receiver) = observation_channel(NonZeroUsize::new(capacity).unwrap());
    let options = CallOptions {
        asynchronous: false,
        lifecycle: lifecycle_binding,
        observers: Some(sender),
    };
    (options, move || {
        std::iter::from_fn(|| receiver.try_recv().ok()).collect()
    })
}

#[rstest]
#[case::success(OpScript::AwaitAnswer)]
#[case::failure(OpScript::AwaitFailure)]
fn asynchronous_host_operations_resume_the_same_machine(
    #[from(interpreter)] _python: Interpreter,
    #[case] op: OpScript,
) {
    Python::attach(|py| {
        let machine: MachineFactory = Box::new(|_| {
            CallMachine::<Synthetic>::new(None, |host| {
                Box::pin(async move { host.services.call(|reply| ("read", reply)).await })
            })
        });
        let (result, log) = run_scripted(py, machine, op, HookScript::Plain, true);
        match op {
            OpScript::AwaitAnswer => {
                assert_eq!(result.unwrap().extract::<String>(py).unwrap(), "awaited")
            }
            OpScript::AwaitFailure => {
                assert!(result.unwrap_err().is_instance_of::<PyLookupError>(py));
                assert_eq!(
                    log.iter()
                        .filter(|entry| entry.starts_with("failed:"))
                        .count(),
                    1
                );
                assert!(!log.iter().any(|entry| entry.starts_with("succeeded:")));
            }
            _ => unreachable!(),
        }
    });
}

#[rstest]
fn composed_hooks_share_prepared_arguments_and_one_execution(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let log = Log::default();
        let chain = HookChain::new()
            .with(hooks(&log, HookScript::ObserveArguments))
            .with(hooks(&log, HookScript::RewriteArguments))
            .with(hooks(&log, HookScript::ReplaceResponse));
        let result = run_call(
            py,
            move |_, _, request| Ok(success_machine()(request)),
            binding(&log, OpScript::Answer),
            chain,
            PyDict::new(py).unbind(),
            call_options(asynchronous),
        );
        if asynchronous {
            assert!(log.entries().is_empty());
        }
        let response = settled(py, result, asynchronous).unwrap();
        assert_eq!(response.extract::<String>(py).unwrap(), "replaced");
        let entries = log.entries();
        assert!(entries.iter().any(|entry| entry == "adopted:hook"));
        assert_eq!(
            entries.iter().filter(|entry| *entry == "project").count(),
            1
        );
        assert_eq!(
            entries.iter().filter(|entry| *entry == "op:sign").count(),
            1
        );
        assert_eq!(
            entries
                .iter()
                .filter(|entry| *entry == "succeeded:replaced")
                .count(),
            3
        );
        assert!(!entries.iter().any(|entry| entry.starts_with("failed:")));
    });
}

#[rstest]
#[case::draining(4, false)]
#[case::full(1, false)]
#[case::closed(4, true)]
fn observation_delivery_preserves_callback_order_and_response(
    #[from(interpreter)] _python: Interpreter,
    #[case] capacity: usize,
    #[case] closed: bool,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let (expected, expected_log) = run_scripted(
            py,
            success_machine(),
            OpScript::Answer,
            HookScript::ReplaceResponse,
            asynchronous,
        );
        let (sender, mut receiver) = observation_channel(NonZeroUsize::new(capacity).unwrap());
        if closed {
            receiver.close();
        }
        let log = Log::default();
        let (actual, actual_log) = run_composed(
            py,
            success_machine(),
            binding(&log, OpScript::Answer),
            hooks(&log, HookScript::ReplaceResponse),
            std::convert::identity,
            CallOptions {
                asynchronous,
                lifecycle: lifecycle_binding,
                observers: Some(sender.clone()),
            },
        );
        assert_eq!(
            actual.unwrap().extract::<String>(py).unwrap(),
            expected.unwrap().extract::<String>(py).unwrap()
        );
        assert_eq!(actual_log, expected_log);
        let events: Vec<_> = std::iter::from_fn(|| receiver.try_recv().ok()).collect();
        assert_eq!(events.len() as u64 + sender.dropped_events(), 3);
        if !closed && capacity >= 3 {
            let [
                CallEvent::Started { start_time },
                CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }),
                CallEvent::Succeeded { timing, .. },
            ] = events.as_slice()
            else {
                panic!("expected start, provider response and success: {events:?}");
            };
            assert_eq!(raw.body, "raw");
            assert_eq!(*start_time, timing.start_time);
            assert!(timing.end_time >= timing.start_time);
        }
    });
}

#[rstest]
#[case::preparation(HookScript::FailBegin, false, 1)]
#[case::transformation(HookScript::FailAfterSuccess, false, 1)]
#[case::terminal_cancellation(HookScript::CancelTerminal, true, 0)]
#[case::terminal_failure(HookScript::FailTerminal, false, 0)]
fn a_terminal_hook_failure_is_reported_once_and_never_replays_callbacks(
    #[from(interpreter)] _python: Interpreter,
    #[case] script: HookScript,
    #[case] cancelled: bool,
    #[case] failure_callbacks: usize,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let (options, mut events) = observed(4);
        let log = Log::default();
        let (result, log) = run_composed(
            py,
            success_machine(),
            binding(&log, OpScript::Answer),
            hooks(&log, script),
            std::convert::identity,
            CallOptions {
                asynchronous,
                ..options
            },
        );
        let error = result.unwrap_err();
        assert_eq!(error.is_instance_of::<CancelledError>(py), cancelled);
        let events = events();
        assert!(matches!(events.first(), Some(CallEvent::Started { .. })));
        assert_eq!(
            matches!(events.last(), Some(CallEvent::Cancelled { .. })),
            cancelled
        );
        assert_eq!(
            matches!(
                events.last(),
                Some(CallEvent::Failed {
                    origin: FailureOrigin::Host,
                    ..
                })
            ),
            !cancelled
        );
        let terminal = events.iter().filter(|event| {
            matches!(
                event,
                CallEvent::Failed { .. }
                    | CallEvent::Succeeded { .. }
                    | CallEvent::Cancelled { .. }
            )
        });
        assert_eq!(terminal.count(), 1);
        assert_eq!(
            log.iter()
                .filter(|entry| entry.starts_with("failed:"))
                .count(),
            failure_callbacks
        );
        assert!(
            !log.iter()
                .any(|entry| entry.starts_with("succeeded:") || entry == "cancelled")
        );
    });
}

#[rstest]
#[case::preparation(HookScript::FailBegin, OpScript::Answer)]
#[case::native_decode(HookScript::Plain, OpScript::RejectRequestNatively)]
#[case::python_decode(HookScript::Plain, OpScript::RaiseRequestPython)]
fn startup_failures_release_the_factory_without_constructing_a_machine(
    #[from(interpreter)] _python: Interpreter,
    #[case] script: HookScript,
    #[case] op: OpScript,
    #[values(false, true)] asynchronous: bool,
) {
    struct Release(Arc<AtomicBool>);
    impl Drop for Release {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    Python::attach(|py| {
        let constructed = Arc::new(AtomicBool::new(false));
        let did_construct = constructed.clone();
        let released = Arc::new(AtomicBool::new(false));
        let release = Release(released.clone());
        let machine: MachineFactory = Box::new(move |request| {
            let _release = release;
            did_construct.store(true, Ordering::SeqCst);
            success_machine()(request)
        });
        let (result, log) = run_scripted(py, machine, op, script, asynchronous);
        assert!(result.is_err());
        assert!(!constructed.load(Ordering::SeqCst));
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(
            log.iter()
                .filter(|event| event.starts_with("failed:"))
                .count(),
            1
        );
        assert_eq!(
            log.iter().filter(|event| *event == "adapter.close").count(),
            1
        );
        assert_eq!(log.iter().filter(|event| *event == "host.close").count(), 1);
    });
}

#[rstest]
fn success_runs_every_step_in_order_and_returns_the_public_response(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let (result, log) = run_scripted(
            py,
            success_machine(),
            OpScript::Answer,
            HookScript::Plain,
            asynchronous,
        );
        assert_eq!(
            result.unwrap().extract::<String>(py).unwrap(),
            "project:1|sign|rewritten"
        );
        assert_eq!(
            log,
            [
                "started",
                "begin",
                "project",
                "op:sign",
                "before_provider_request",
                "response:raw",
                "complete",
                "after_success",
                "succeeded:project:1|sign|rewritten",
                "adapter.close",
                "host.close",
            ]
        );
    });
}

#[rstest]
fn explicitly_closing_a_stream_dispatches_success_for_delivered_chunks(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let log = Log::default();
        let (options, mut events) = observed(3);
        let stream = open_stream(
            py,
            hooks(&log, HookScript::Plain),
            CallOptions {
                asynchronous,
                ..options
            },
        );
        if asynchronous {
            for method in ["__anext__", "aclose", "aclose"] {
                let stop = stream
                    .call_method0(method)
                    .unwrap()
                    .call_method1("send", (py.None(),))
                    .unwrap_err();
                assert!(stop.is_instance_of::<PyStopIteration>(py));
            }
        } else {
            assert_eq!(
                stream
                    .call_method0("__next__")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "first"
            );
            stream.call_method0("close").unwrap();
            stream.call_method0("close").unwrap();
        }
        assert!(matches!(
            events().as_slice(),
            [CallEvent::Started { .. }, CallEvent::Succeeded { .. }]
        ));
        assert_eq!(
            log.entries(),
            [
                "started",
                "begin",
                "opened",
                "delivered",
                "succeeded:None",
                "adapter.close"
            ]
        );
    });
}

#[rstest]
fn a_stream_carries_its_head_before_the_first_chunk(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let log = Log::default();
        let stream = open_stream(
            py,
            hooks(&log, HookScript::Plain),
            call_options(asynchronous),
        );
        let hidden: std::collections::HashMap<String, std::collections::HashMap<String, String>> =
            stream.getattr("head").unwrap().extract().unwrap();
        assert_eq!(
            hidden["additional_headers"],
            std::collections::HashMap::from([("request-id".to_string(), "req_1".to_string())])
        );
        assert_eq!(log.entries(), ["started", "begin", "opened"]);
        assert_eq!(read_all(py, &stream, asynchronous), ["first", "second"]);
    });
}

struct Classification {
    provider_fails: bool,
    op: OpScript,
    classifier_fails: bool,
    message: &'static str,
    native_context: Option<&'static str>,
    steps: &'static [&'static str],
}

#[rstest]
#[case::provider_failure(Classification {
    provider_fails: true,
    op: OpScript::Answer,
    classifier_fails: false,
    message: "classified: provider exploded",
    native_context: None,
    steps: &["project", "classify:provider exploded"],
})]
#[case::host_op_rejection(Classification {
    provider_fails: false,
    op: OpScript::RejectNatively,
    classifier_fails: false,
    message: "classified: op rejected",
    native_context: None,
    steps: &["project", "op:sign", "classify:op rejected"],
})]
#[case::host_op_python_exception(Classification {
    provider_fails: false,
    op: OpScript::RaisePython,
    classifier_fails: false,
    message: "op failed",
    native_context: None,
    steps: &["project", "op:sign"],
})]
#[case::failing_classifier(Classification {
    provider_fails: true,
    op: OpScript::Answer,
    classifier_fails: true,
    message: "classifier failed",
    native_context: Some("provider exploded"),
    steps: &["project", "classify:provider exploded"],
})]
fn a_native_failure_is_classified_once_and_a_python_exception_is_raised_as_is(
    #[from(interpreter)] _python: Interpreter,
    #[case] case: Classification,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let Classification {
            provider_fails,
            op,
            classifier_fails,
            message,
            native_context,
            steps,
        } = case;
        let log = Log::default();
        let binding = if classifier_fails {
            binding(&log, op).with_failing_classifier()
        } else {
            binding(&log, op)
        };
        let machine = if provider_fails {
            failing_machine()
        } else {
            success_machine()
        };
        let (result, log) = run_composed(
            py,
            machine,
            binding,
            hooks(&log, HookScript::Plain),
            std::convert::identity,
            call_options(asynchronous),
        );
        let error = result.unwrap_err();
        assert_eq!(error.value(py).to_string(), message);
        match native_context {
            Some(context) => {
                assert!(error.is_instance_of::<PyTypeError>(py));
                let chained = error.context(py).unwrap();
                assert!(chained.is_instance_of::<PyRuntimeError>(py));
                assert_eq!(chained.value(py).to_string(), context);
            }
            None => assert!(error.is_instance_of::<PyValueError>(py)),
        }
        let expected: Vec<String> = ["started", "begin"]
            .iter()
            .chain(steps)
            .map(|step| step.to_string())
            .chain([
                format!("failed:Call:{message}"),
                "adapter.close".into(),
                "host.close".into(),
            ])
            .collect();
        assert_eq!(log, expected);
    });
}

#[rstest]
fn begin_failures_are_host_failures_without_provider_mapping(
    #[from(interpreter)] _python: Interpreter,
) {
    Python::attach(|py| {
        let (result, log) = run_scripted(
            py,
            success_machine(),
            OpScript::Answer,
            HookScript::FailBegin,
            false,
        );
        assert_eq!(result.unwrap_err().value(py).to_string(), "begin failed");
        assert_eq!(
            log,
            [
                "started",
                "begin",
                "failed:Host:begin failed",
                "adapter.close",
                "host.close"
            ]
        );
    });
}

enum ArgumentPolicy {
    Inherit,
    Reject(Py<PyBaseException>),
}

impl CallHooks<PythonRuntime> for ArgumentPolicy {
    fn arguments_prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| match self {
            Self::Inherit => arguments.bind(py).set_item("api_key", "inherited"),
            Self::Reject(error) => Err(PyErr::from_value(error.bind(py).clone().into_any())),
        })
    }
}

impl PythonOwned for ArgumentPolicy {
    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Inherit => Ok(()),
            Self::Reject(error) => visit.call(error),
        }
    }
}

#[rstest]
fn resource_setup_uses_prepared_arguments_and_failures_are_terminal(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
    #[values(false, true)] fail_setup: bool,
) {
    Python::attach(|py| {
        let log = Log::default();
        let setup_log = log.clone();
        let error = PyValueError::new_err("setup failed").into_value(py);
        let setup_error = error.clone_ref(py);
        let arguments = PyDict::new(py);
        arguments.set_item("api_key", "original").unwrap();
        let result = run_call(
            py,
            move |py, prepared, request| {
                let source: String = prepared.get_item("prepared")?.unwrap().extract()?;
                let key: String = prepared.get_item("api_key")?.unwrap().extract()?;
                setup_log.push(format!("setup:{source}:{key}"));
                if fail_setup {
                    return Err(PyErr::from_value(setup_error.into_bound(py).into_any()));
                }
                Ok(success_machine()(request))
            },
            binding(&log, OpScript::Answer),
            HookChain::new()
                .with(hooks(&log, HookScript::RewriteArguments))
                .with(ArgumentPolicy::Inherit),
            arguments.clone().unbind(),
            call_options(asynchronous),
        );
        if asynchronous {
            assert!(log.entries().is_empty());
        }
        let settled = settled(py, result, asynchronous);
        assert_eq!(
            arguments
                .get_item("api_key")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "original"
        );
        assert_eq!(
            &log.entries()[..4],
            ["started", "begin", "project", "setup:hook:inherited"]
        );
        if fail_setup {
            assert!(settled.unwrap_err().value(py).is(error.bind(py)));
            assert_eq!(
                log.entries(),
                [
                    "started",
                    "begin",
                    "project",
                    "setup:hook:inherited",
                    "failed:Host:setup failed",
                    "adapter.close",
                    "host.close",
                ]
            );
        } else {
            assert!(settled.is_ok());
            assert!(
                log.entries()
                    .iter()
                    .any(|entry| entry.starts_with("succeeded:"))
            );
        }
    });
}

#[rstest]
fn closing_an_unstarted_call_never_initializes_resources(
    #[from(interpreter)] _python: Interpreter,
) {
    Python::attach(|py| {
        let log = Log::default();
        let setup_log = log.clone();
        let pending = run_call(
            py,
            move |_, _, request| {
                setup_log.push("setup");
                Ok(success_machine()(request))
            },
            binding(&log, OpScript::Answer),
            hooks(&log, HookScript::Plain),
            PyDict::new(py).unbind(),
            call_options(true),
        )
        .unwrap();
        assert!(log.entries().is_empty());
        pending.call_method0(py, "close").unwrap();
        drop(pending);
        assert_eq!(log.entries(), ["adapter.close", "host.close"]);
    });
}

#[rstest]
fn an_argument_policy_rejection_is_the_callers_error_and_the_machine_never_starts(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let raised = PyValueError::new_err("over budget").into_value(py);
        let log = Log::default();
        let (result, log) = run_composed(
            py,
            success_machine(),
            binding(&log, OpScript::Answer),
            hooks(&log, HookScript::Plain),
            |hooks| {
                HookChain::new()
                    .with(hooks)
                    .with(ArgumentPolicy::Reject(raised.clone_ref(py)))
            },
            call_options(asynchronous),
        );
        assert!(result.unwrap_err().value(py).is(raised.bind(py)));
        assert_eq!(
            log,
            [
                "started",
                "begin",
                "failed:Host:over budget",
                "adapter.close",
                "host.close"
            ]
        );
    });
}

#[rstest]
fn the_host_projects_from_the_keyword_view_the_argument_policy_rewrote(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let log = Log::default();
        let (result, _) = run_composed(
            py,
            success_machine(),
            binding(&log, OpScript::Answer),
            hooks(&log, HookScript::Plain),
            |hooks| HookChain::new().with(hooks).with(ArgumentPolicy::Inherit),
            call_options(asynchronous),
        );
        assert_eq!(
            result.unwrap().extract::<String>(py).unwrap(),
            "project:2|sign|rewritten"
        );
    });
}

#[rstest]
fn a_failure_while_finalizing_fails_the_call_instead_of_succeeding(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let (result, log) = run_scripted(
            py,
            success_machine(),
            OpScript::Answer,
            HookScript::FailAfterSuccess,
            asynchronous,
        );
        assert_eq!(
            result.unwrap_err().value(py).to_string(),
            "after_success failed"
        );
        assert_eq!(
            &log[log.len() - 4..],
            [
                "after_success",
                "failed:Host:after_success failed",
                "adapter.close",
                "host.close"
            ]
        );
        assert!(!log.iter().any(|entry| entry.starts_with("succeeded")));
    });
}

#[rstest]
fn cancellation_ends_the_call_without_terminal_dispatch(
    #[from(interpreter)] _python: Interpreter,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let (result, log) = run_scripted(
            py,
            success_machine(),
            OpScript::CancelRequest,
            HookScript::Plain,
            asynchronous,
        );
        let error = result.unwrap_err();
        assert!(!error.is_instance_of::<PyException>(py));
        assert_eq!(
            log,
            ["started", "begin", "project", "adapter.close", "host.close"]
        );
    });
}
