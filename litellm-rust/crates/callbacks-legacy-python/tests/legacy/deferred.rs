use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::rstest;

use super::{PendingLogging, PendingSuccess};
use crate::PythonLogger;
use crate::test_support::{local, namespace, run};

/// A deferred success for the namespace's `logger` and `response`, bound as `pending`.
fn defer<'py>(py: Python<'py>, script: &CStr) -> Bound<'py, PyDict> {
    let locals = namespace(py, c"response = object()");
    run(py, &locals, script);
    let pending = Py::new(
        py,
        PendingLogging {
            pending: Some(PendingSuccess {
                logger: PythonLogger::new(local(&locals, "logger").unbind()),
                response: Some(local(&locals, "response").unbind()),
                start: py.None(),
                end: Some(py.None()),
            }),
        },
    )
    .unwrap();
    locals.set_item("pending", pending).unwrap();
    locals
}

#[test]
fn release_enqueues_the_success_once_in_the_releasing_context() {
    Python::initialize();
    Python::attach(|py| {
        let locals = defer(
            py,
            c"
from contextvars import ContextVar

marker = ContextVar('marker', default='unset')
observed = []

def on_enqueue(coroutine):
    observed.append(marker.get())
    pending.release(True)

logger.on_enqueue = on_enqueue
",
        );
        run(
            py,
            &locals,
            c"
marker.set('release')
pending.release(True)
pending.release(True)
assert observed == ['release'], observed
assert logger.names() == ['async_success_handler', 'enqueued'], logger.calls
assert logger.calls[0][1] is response
",
        );
    });
}

#[test]
fn a_blocked_release_drops_the_success_for_good() {
    Python::initialize();
    Python::attach(|py| {
        let locals = defer(py, c"");
        run(
            py,
            &locals,
            c"
pending.release(False)
pending.release(True)
assert logger.calls == [], logger.calls
",
        );
    });
}

#[rstest]
#[case::ordinary_error(c"RuntimeError('queue full')", false)]
#[case::cancellation(c"asyncio.CancelledError()", true)]
fn a_failed_enqueue_closes_the_coroutine_and_is_never_replayed(
    #[case] failure: &CStr,
    #[case] propagates: bool,
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = defer(
            py,
            c"
import asyncio

def on_enqueue(coroutine):
    raise failure

logger.on_enqueue = on_enqueue
",
        );
        locals
            .set_item("failure", py.eval(failure, None, Some(&locals)).unwrap())
            .unwrap();
        let released = local(&locals, "pending").call_method1("release", (true,));
        match released {
            Ok(_) => assert!(!propagates),
            Err(error) => {
                assert!(propagates);
                assert!(error.value(py).is(local(&locals, "failure")));
            }
        }
        locals.set_item("propagates", propagates).unwrap();
        run(
            py,
            &locals,
            c"
pending.release(True)
assert logger.names() == ['async_success_handler', 'enqueued', 'closed'], logger.calls
assert unraisable_from(logger) == ([] if propagates else [failure])
",
        );
    });
}

#[test]
fn an_unreleased_success_does_not_keep_its_logger_alive() {
    Python::initialize();
    Python::attach(|py| {
        let locals = defer(py, c"");
        run(
            py,
            &locals,
            c"
import gc
import weakref

logger.pending = pending
reference = weakref.ref(logger)
del logger, pending
gc.collect()
assert reference() is None
",
        );
    });
}
