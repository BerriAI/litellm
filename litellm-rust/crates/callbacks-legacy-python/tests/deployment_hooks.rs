use std::ffi::CStr;

use litellm_host::event::{FailureOrigin, Timing};
use litellm_host_python::{LifecycleEvent, LifecycleStep, PythonLifecycle};
use pyo3::exceptions::asyncio::CancelledError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::rstest;

use super::LegacyLogging;
use crate::test_support::{legacy_call, local, namespace, run};

const CALL: &CStr = c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
kwargs = {'logger': logger, 'document': document}
";

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

fn begin<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    asynchronous: bool,
) -> (LegacyLogging, LifecycleStep) {
    let mut logging = legacy_call(py, locals, asynchronous);
    let kwargs = local(locals, "kwargs")
        .cast_into::<PyDict>()
        .unwrap()
        .unbind();
    let step = logging.begin(py, kwargs, 0.0).unwrap();
    (logging, step)
}

fn arguments<'py>(py: Python<'py>, step: LifecycleStep) -> Bound<'py, PyDict> {
    let LifecycleStep::Arguments(arguments) = step else {
        panic!("expected the prepared arguments");
    };
    arguments.into_bound(py)
}

fn awaits_deployment_hook(step: &LifecycleStep) -> bool {
    matches!(step, LifecycleStep::Await(_))
}

#[rstest]
#[case::synchronous(false)]
#[case::asynchronous(true)]
fn deployment_pre_call_hook_runs_only_for_asynchronous_calls(#[case] asynchronous: bool) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, CALL);
        let (_, step) = begin(py, &locals, asynchronous);
        assert_eq!(awaits_deployment_hook(&step), asynchronous);
        let names: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(names.contains(&"pre_hook".to_string()), asynchronous);
    });
}

#[test]
fn kwargs_returned_by_the_pre_call_hook_are_what_the_call_prepares() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,YWJj'}
replacement = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,ZWRpdGVk'}
kwargs = {'logger': logger, 'document': document}
replaced_kwargs = {'logger': logger, 'document': replacement, 'pages': [0]}
",
        );
        let (mut logging, step) = begin(py, &locals, true);
        assert!(awaits_deployment_hook(&step));
        let step = logging
            .resume(py, Ok(local(&locals, "replaced_kwargs").unbind()))
            .unwrap();
        locals.set_item("prepared", arguments(py, step)).unwrap();
        run(
            py,
            &locals,
            c"
assert prepared['document'] is replacement
assert prepared['pages'] is replaced_kwargs['pages']
assert prepared['litellm_logging_obj'] is logger
assert 'litellm_logging_obj' not in replaced_kwargs
[checked] = [value for name, value in logger.calls if name == 'check_limits']
assert checked is prepared
",
        );
    });
}

#[rstest]
#[case::synchronous(false)]
#[case::asynchronous(true)]
fn a_keyword_the_bridge_never_reads_reaches_every_reader_as_the_callers_object(
    #[case] asynchronous: bool,
) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
opaque = object()
hooked = []
logger.hooks = {'pre': lambda kwargs: hooked.append(kwargs['vendor_extension']) or kwargs}
kwargs = {'logger': logger, 'vendor_extension': opaque}
",
        );
        let (mut logging, step) = begin(py, &locals, asynchronous);
        let step = match step {
            LifecycleStep::Await(hook_result) => logging.resume(py, Ok(hook_result)).unwrap(),
            step => step,
        };
        locals.set_item("prepared", arguments(py, step)).unwrap();
        locals.set_item("asynchronous", asynchronous).unwrap();
        run(
            py,
            &locals,
            c"
assert prepared['vendor_extension'] is opaque
[checked] = [value for name, value in logger.calls if name == 'check_limits']
assert checked['vendor_extension'] is opaque
assert hooked == ([opaque] if asynchronous else []), hooked
",
        );
    });
}

#[test]
fn response_returned_by_the_post_call_hook_is_finalized_and_returned() {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
kwargs = {'logger': logger}
response = object()
replacement = object()
logger.hooks = {'pre': lambda kwargs: kwargs}
",
        );
        let (mut logging, _) = begin(py, &locals, true);
        logging
            .resume(py, Ok(local(&locals, "kwargs").unbind()))
            .unwrap();
        let step = logging
            .after_success(py, local(&locals, "response").unbind(), TIMING)
            .unwrap();
        assert!(awaits_deployment_hook(&step));
        let step = logging
            .resume(py, Ok(local(&locals, "replacement").unbind()))
            .unwrap();
        let LifecycleStep::Response(returned) = step else {
            panic!("expected the finalized response");
        };
        assert!(returned.bind(py).is(local(&locals, "replacement")));
        run(
            py,
            &locals,
            c"
[finalized] = [value for name, value in logger.calls if name == 'finalize']
assert finalized is replacement
",
        );
    });
}

#[rstest]
#[case::pre_call(false)]
#[case::post_call(true)]
fn cancelling_a_deployment_hook_ends_the_call_with_that_cancellation(#[case] post_call: bool) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(py, c"kwargs = {'logger': logger}\nresponse = object()");
        let (mut logging, _) = begin(py, &locals, true);
        if post_call {
            logging
                .resume(py, Ok(local(&locals, "kwargs").unbind()))
                .unwrap();
            logging
                .after_success(py, local(&locals, "response").unbind(), TIMING)
                .unwrap();
        }
        let cancellation = CancelledError::new_err("cancelled");
        let cancelled = cancellation.value(py).clone();
        let error = logging.resume(py, Err(cancellation)).err().unwrap();
        assert!(error.value(py).is(&cancelled));
        let names: Vec<String> = local(&locals, "logger")
            .call_method0("names")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!names.iter().any(|name| name.contains("handler")));
    });
}

#[rstest]
#[case::hook_completed(false)]
#[case::hook_cancelled(true)]
fn failure_callbacks_run_after_the_failure_hook_however_it_ends(#[case] cancelled: bool) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"kwargs = {'logger': logger}\nfailure = ValueError('provider')",
        );
        let (mut logging, _) = begin(py, &locals, true);
        logging
            .resume(py, Ok(local(&locals, "kwargs").unbind()))
            .unwrap();
        let failure = PyErr::from_value(local(&locals, "failure"));
        let failed = LifecycleEvent::Failed {
            timing: TIMING,
            origin: FailureOrigin::Call,
            error: &failure,
        };
        let step = logging.emit(py, failed).unwrap();
        assert!(awaits_deployment_hook(&step));
        let hook_result = if cancelled {
            Err(CancelledError::new_err("cancelled"))
        } else {
            Ok(py.None())
        };
        assert!(matches!(
            logging.resume(py, hook_result).unwrap(),
            LifecycleStep::Await(_)
        ));
        run(
            py,
            &locals,
            c"
assert logger.names()[-3:] == ['failure_hook', 'failure_handler', 'async_failure_handler'], logger.calls
assert all(value is failure for name, value in logger.calls if name.endswith('_handler'))
",
        );
    });
}

#[rstest]
#[case::synchronous(false)]
#[case::asynchronous(true)]
fn a_limit_rejected_before_the_call_surfaces_as_the_callers_error(#[case] asynchronous: bool) {
    Python::initialize();
    Python::attach(|py| {
        let locals = namespace(
            py,
            c"
class BudgetExceeded(Exception):
    pass

rejection = BudgetExceeded('over budget')

class LimitedLogger(StubLogger):
    def check_limits(self, arguments):
        raise rejection

logger = LimitedLogger()
logger.hooks = {'pre': lambda kwargs: kwargs}
kwargs = {'logger': logger}
",
        );
        let mut logging = legacy_call(py, &locals, asynchronous);
        let kwargs = local(&locals, "kwargs")
            .cast_into::<PyDict>()
            .unwrap()
            .unbind();
        let result = logging.begin(py, kwargs, 0.0).and_then(|step| match step {
            LifecycleStep::Await(_) => logging.resume(py, Ok(local(&locals, "kwargs").unbind())),
            step => Ok(step),
        });
        let error = result.err().unwrap();
        assert!(error.value(py).is(local(&locals, "rejection")));
    });
}
