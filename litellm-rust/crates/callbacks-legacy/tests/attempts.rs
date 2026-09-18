//! The legacy contract under a loop of attempts. Python runs every attempt as its own
//! `@client` call, so each later attempt sets up again with a fresh call id and the shared
//! trace id, each failure reaches the failure families when the loop reports it, and the
//! terminal failure that follows the last reported attempt dispatches nothing more.

use std::ffi::CStr;

use litellm_callbacks::event::{AttemptInfo, CallEvent, FailureOrigin, Timing};
use litellm_callbacks::failure::FailureClass;
use litellm_host_python::{AdapterStep, CallbackAdapter, PublicValue};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::LegacyLogging;
use crate::test_support::{legacy_call, local, namespace};

/// `setup` hands out a new logger per call, as `function_setup` does, and each logger
/// remembers the keywords it was set up with.
const PER_ATTEMPT_SETUP: &CStr = c"
loggers = []

class AttemptLogger(StubLogger):
    def __init__(self, kwargs):
        super().__init__()
        self.setup_kwargs = dict(kwargs)
        loggers.append(self)

kwargs = {
    'model': 'm',
    'logger': logger,
    'logger_factory': AttemptLogger,
    'litellm_trace_id': 'caller-trace',
}
failure = RuntimeError('attempt failed')
response = object()
";

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

fn attempt(index: u32) -> AttemptInfo {
    AttemptInfo {
        trace_id: "router-trace".into(),
        index,
        group: 0,
        deployment: 1,
    }
}

/// Answers every deployment-hook await with `answer` until the adapter settles.
fn settle(
    py: Python<'_>,
    logging: &mut LegacyLogging,
    mut step: AdapterStep,
    answer: &Bound<'_, PyAny>,
) -> AdapterStep {
    while let AdapterStep::Await(_) = step {
        step = logging.resume(py, Ok(answer.clone().unbind())).unwrap();
    }
    step
}

fn started(
    py: Python<'_>,
    locals: &Bound<'_, PyDict>,
    logging: &mut LegacyLogging,
    index: u32,
) -> AdapterStep {
    let step = logging
        .emit(
            py,
            &CallEvent::AttemptStarted {
                attempt: attempt(index),
            },
            None,
        )
        .unwrap();
    settle(py, logging, step, &local(locals, "kwargs"))
}

fn begun(py: Python<'_>, locals: &Bound<'_, PyDict>, asynchronous: bool) -> LegacyLogging {
    let mut logging = legacy_call(py, locals, asynchronous);
    let kwargs = local(locals, "kwargs");
    let step = logging
        .begin(
            py,
            kwargs.clone().cast_into::<PyDict>().unwrap().unbind(),
            0.0,
        )
        .unwrap();
    assert!(matches!(
        settle(py, &mut logging, step, &kwargs),
        AdapterStep::Arguments(_)
    ));
    logging
}

fn failed(
    py: Python<'_>,
    locals: &Bound<'_, PyDict>,
    logging: &mut LegacyLogging,
    index: u32,
) -> AdapterStep {
    let failure = PyErr::from_value(local(locals, "failure"));
    let step = logging
        .attempt_failed(py, &attempt(index), FailureClass::RateLimited, &failure)
        .unwrap();
    settle(py, logging, step, &py.None().into_bound(py))
}

fn logger_names(locals: &Bound<'_, PyDict>, index: usize) -> Vec<String> {
    local(locals, "loggers")
        .get_item(index)
        .unwrap()
        .call_method0("names")
        .unwrap()
        .extract()
        .unwrap()
}

fn setup_kwargs<'py>(locals: &Bound<'py, PyDict>, index: usize) -> Bound<'py, PyDict> {
    local(locals, "loggers")
        .get_item(index)
        .unwrap()
        .getattr("setup_kwargs")
        .unwrap()
        .cast_into()
        .unwrap()
}

#[test]
fn a_later_attempt_sets_up_again_with_the_trace_id_and_without_the_previous_call() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PER_ATTEMPT_SETUP);
        let mut logging = begun(py, &locals, true);
        assert!(matches!(
            started(py, &locals, &mut logging, 0),
            AdapterStep::Done
        ));
        assert_eq!(local(&locals, "loggers").len().unwrap(), 1);

        let AdapterStep::Arguments(arguments) = started(py, &locals, &mut logging, 1) else {
            panic!("a later attempt hands the driver its own keyword view")
        };

        assert_eq!(local(&locals, "loggers").len().unwrap(), 2);
        assert_eq!(logger_names(&locals, 0), ["restore"]);
        let second = setup_kwargs(&locals, 1);
        assert!(!second.contains("litellm_call_id").unwrap());
        assert!(!second.contains("litellm_logging_obj").unwrap());
        assert_eq!(
            second
                .get_item("litellm_trace_id")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "router-trace"
        );
        assert!(
            arguments
                .bind(py)
                .get_item("litellm_logging_obj")
                .unwrap()
                .unwrap()
                .is(local(&locals, "loggers").get_item(1).unwrap())
        );
    });
}

#[test]
fn an_attempt_failure_runs_the_hook_and_both_failure_families_on_that_attempts_logger() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PER_ATTEMPT_SETUP);
        let mut logging = begun(py, &locals, true);
        assert!(matches!(
            started(py, &locals, &mut logging, 1),
            AdapterStep::Arguments(_)
        ));

        assert!(matches!(
            failed(py, &locals, &mut logging, 1),
            AdapterStep::Done
        ));

        assert_eq!(
            logger_names(&locals, 1),
            ["failure_handler", "async_failure_handler"]
        );
        assert!(
            !logger_names(&locals, 0)
                .iter()
                .any(|name| name.contains("failure")),
            "the earlier attempt's logger saw nothing"
        );
        let hooks: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert!(hooks.contains(&"failure_hook".to_string()));
    });
}

#[test]
fn the_terminal_failure_after_a_reported_attempt_dispatches_nothing_more() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PER_ATTEMPT_SETUP);
        let mut logging = begun(py, &locals, false);
        assert!(matches!(
            failed(py, &locals, &mut logging, 0),
            AdapterStep::Done
        ));
        let before = logger_names(&locals, 0);
        assert_eq!(before, ["failure_handler"]);
        let failure = PyErr::from_value(local(&locals, "failure"));

        let step = logging
            .emit(
                py,
                &CallEvent::Failed {
                    timing: TIMING,
                    origin: FailureOrigin::Call,
                },
                Some(PublicValue::Error(&failure)),
            )
            .unwrap();

        assert!(matches!(step, AdapterStep::Done));
        assert_eq!(logger_names(&locals, 0), before);
    });
}

#[test]
fn success_after_a_failed_attempt_reaches_only_the_new_attempts_logger() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, PER_ATTEMPT_SETUP);
        let mut logging = begun(py, &locals, false);
        failed(py, &locals, &mut logging, 0);
        assert!(matches!(
            started(py, &locals, &mut logging, 1),
            AdapterStep::Arguments(_)
        ));
        let response = local(&locals, "response").unbind();

        let step = logging
            .emit(
                py,
                &CallEvent::Succeeded { timing: TIMING },
                Some(PublicValue::Response(&response)),
            )
            .unwrap();

        assert!(matches!(step, AdapterStep::Done));
        assert_eq!(logger_names(&locals, 1), ["submit"]);
        assert_eq!(logger_names(&locals, 0), ["failure_handler", "restore"]);
    });
}
