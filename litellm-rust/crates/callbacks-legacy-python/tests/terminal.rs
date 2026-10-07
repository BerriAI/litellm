use std::ffi::CStr;

use litellm_host::event::{FailureOrigin, Timing};
use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::rstest;

use super::LegacyLogging;
use crate::PythonLogger;
use crate::test_support::{legacy_call, local, namespace, run};

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

fn logged(py: Python<'_>, locals: &Bound<'_, PyDict>, asynchronous: bool) -> LegacyLogging {
    LegacyLogging {
        logger: Some(PythonLogger::new(local(locals, "logger").unbind())),
        ..legacy_call(py, locals, asynchronous)
    }
}

fn succeed(
    py: Python<'_>,
    locals: &Bound<'_, PyDict>,
    logging: &mut LegacyLogging,
) -> LifecycleStep {
    let response = local(locals, "response").unbind();
    logging
        .emit(
            py,
            LifecycleEvent::Succeeded {
                timing: TIMING,
                response: &response,
            },
        )
        .unwrap()
}

fn fail(py: Python<'_>, locals: &Bound<'_, PyDict>, logging: &mut LegacyLogging) -> LifecycleStep {
    let failure = PyErr::from_value(local(locals, "failure"));
    logging
        .emit(
            py,
            LifecycleEvent::Failed {
                timing: TIMING,
                origin: FailureOrigin::Host,
                error: &failure,
            },
        )
        .unwrap()
}

#[rstest]
#[case::sync_listened(false, c"", &["submit"])]
#[case::async_listened(
    true,
    c"",
    &["async_success_handler", "enqueued", "sync_success_for_async_call"]
)]
#[case::async_deferred(true, c"logger._defer_async_logging = True", &["sync_success_for_async_call"])]
#[case::async_with_fallbacks(true, c"kwargs = {'fallbacks': ['other']}", &["sync_success_for_async_call"])]
fn success_reaches_the_logging_handlers(
    #[case] asynchronous: bool,
    #[case] script: &CStr,
    #[case] expected: &[&str],
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"response = object()");
        run(py, &locals, script);
        let mut logging = logged(py, &locals, asynchronous);
        assert!(matches!(
            succeed(py, &locals, &mut logging),
            LifecycleStep::Done
        ));
        let names: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(names, expected);
        run(
            py,
            &locals,
            c"
assert all(value is response for name, value in logger.calls if name.endswith('_handler'))
assert hasattr(logger, '_native_pending_logging') == getattr(logger, '_defer_async_logging', False)
",
        );
    });
}

#[rstest]
#[case::synchronous(false, &["failure_handler"])]
#[case::asynchronous(true, &[])]
fn internal_calls_skip_failure_callbacks_only_when_asynchronous(
    #[case] asynchronous: bool,
    #[case] expected: &[&str],
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"failure = ValueError('provider')");
        let mut logging = LegacyLogging {
            internal: true,
            ..logged(py, &locals, asynchronous)
        };
        assert!(matches!(
            fail(py, &locals, &mut logging),
            LifecycleStep::Done
        ));
        let names: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(names, expected);
    });
}

#[test]
fn internal_async_calls_skip_the_async_success_fan_out() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"response = object()");
        let mut logging = LegacyLogging {
            internal: true,
            ..logged(py, &locals, true)
        };
        succeed(py, &locals, &mut logging);
        run(
            py,
            &locals,
            c"assert logger.names() == ['sync_success_for_async_call'], logger.calls",
        );
    });
}

#[test]
fn a_failing_success_callback_is_reported_without_replacing_the_response() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
response = object()
failure = ValueError('terminal diagnostic')

class FailingLogger(StubLogger):
    def handle_sync_success_callbacks_for_async_calls(self, *args):
        raise failure

logger = FailingLogger()
",
        );
        let mut logging = logged(py, &locals, true);
        assert!(matches!(
            succeed(py, &locals, &mut logging),
            LifecycleStep::Done
        ));
        assert!(
            logging
                .response
                .as_ref()
                .unwrap()
                .bind(py)
                .is(local(&locals, "response"))
        );
        run(py, &locals, c"assert unraisable_from(logger) == [failure]");
    });
}

#[rstest]
#[case::sync_listened(false, c"", &["failure_handler"])]
#[case::async_listened(true, c"", &["failure_handler", "async_failure_handler"])]
fn failure_reaches_the_logging_handlers(
    #[case] asynchronous: bool,
    #[case] script: &CStr,
    #[case] expected: &[&str],
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"failure = ValueError('provider')");
        run(py, &locals, script);
        let mut logging = logged(py, &locals, asynchronous);
        let step = fail(py, &locals, &mut logging);
        let awaits_async_handler = expected.contains(&"async_failure_handler");
        assert_eq!(
            matches!(step, LifecycleStep::Await(_)),
            awaits_async_handler
        );
        let names: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(names, expected);
        run(
            py,
            &locals,
            c"assert all(value is failure for name, value in logger.calls if name.endswith('_handler'))",
        );
    });
}

#[test]
fn a_failing_sync_failure_callback_keeps_the_error_and_still_runs_the_async_family() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
failure = ValueError('selected')

class FailingLogger(StubLogger):
    def failure_handler(self, error, trace, start, end):
        self.record('failure_handler', error)
        raise RuntimeError('handler failed')

logger = FailingLogger()
",
        );
        let mut logging = logged(py, &locals, true);
        assert!(matches!(
            fail(py, &locals, &mut logging),
            LifecycleStep::Await(_)
        ));
        assert!(
            logging
                .error
                .as_ref()
                .unwrap()
                .bind(py)
                .is(local(&locals, "failure"))
        );
        run(
            py,
            &locals,
            c"assert logger.names() == ['failure_handler', 'async_failure_handler'], logger.calls",
        );
    });
}

#[rstest]
#[case::completed(None, true)]
#[case::handler_error(Some(false), true)]
#[case::cancelled(Some(true), false)]
fn the_async_failure_handler_ends_the_call_unless_it_was_cancelled(
    #[case] error: Option<bool>,
    #[case] done: bool,
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"failure = ValueError('provider')");
        let mut logging = logged(py, &locals, true);
        fail(py, &locals, &mut logging);
        let result = match error {
            None => Ok(py.None()),
            Some(false) => Err(PyRuntimeError::new_err("handler failed")),
            Some(true) => Err(CancelledError::new_err("cancelled")),
        };
        let expected = result.as_ref().err().map(|error| error.value(py).clone());
        match logging.resume(py, result) {
            Ok(step) => assert!(done && matches!(step, LifecycleStep::Done)),
            Err(propagated) => {
                assert!(!done);
                assert!(propagated.value(py).is(expected.unwrap()));
            }
        }
    });
}

#[test]
fn closing_restores_the_correlation_context_once() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"");
        let mut logging = logged(py, &locals, true);
        logging.close(py);
        logging.close(py);
        run(
            py,
            &locals,
            c"assert logger.names() == ['restore'], logger.calls",
        );
    });
}
