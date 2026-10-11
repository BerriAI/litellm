//! The dispatched handler's body as the native call sees it. Python's async handlers run
//! body hooks between the `@client` wrapper and the provider call: Messages runs
//! `async_pre_request_hook` over the resolved call before the provider transform and the
//! agentic-loop hooks over the complete response before `@client` sees it. Both fan-outs
//! stay in Python, keyed by operation, and are awaited on the caller's task; this hook hands
//! them the resolved request and folds their answers back into the keyword view.

use litellm_host::{hooks::CallHooks, lifecycle::Timing};
use litellm_host_python::{
    PythonOwned, PythonRuntime, PythonTask, effective_py_args, missing_state,
};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{LoggingOperation, PublicCall, python::Body};

pub struct HandlerBody {
    operation: LoggingOperation,
    base: Option<Py<PyDict>>,
    request: Option<Py<PyDict>>,
}

fn runs_body_hooks(operation: LoggingOperation, asynchronous: bool) -> bool {
    matches!(
        (operation, asynchronous),
        (LoggingOperation::Messages, true)
    )
}

impl HandlerBody {
    /// The body hooks for a call, when Python's handler would run them: the async Messages
    /// route today. Every other call keeps its wrapper-only contract.
    pub fn new(
        py: Python<'_>,
        operation: LoggingOperation,
        call: &PublicCall,
        asynchronous: bool,
    ) -> Option<Self> {
        runs_body_hooks(operation, asynchronous).then(|| Self {
            operation,
            base: Some(call.base(py)),
            request: None,
        })
    }

    fn resolved<'py>(
        &self,
        py: Python<'py>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let base = self.base.as_ref().ok_or_else(missing_state)?;
        effective_py_args(base.bind(py), kwargs)
    }
}

/// Lays the keys Python changed over the keyword layer. A key whose value is the very object
/// the resolved request already carried came from the signature base or the caller, so it
/// stays where it was instead of being copied into the layer.
fn write_back(
    before: &Bound<'_, PyDict>,
    replacement: &Bound<'_, PyDict>,
    layer: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for (name, value) in replacement.iter() {
        let unchanged = before
            .get_item(&name)?
            .is_some_and(|previous| previous.is(&value));
        if !unchanged {
            layer.set_item(name, value)?;
        }
    }
    Ok(())
}

impl CallHooks<PythonRuntime> for HandlerBody {
    async fn prepare_request(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let (before, awaitable) = Python::attach(|py| -> PyResult<_> {
            let resolved = self.resolved(py, arguments.bind(py))?;
            let awaitable = Body::PrepareRequest
                .call(py, (<&str>::from(self.operation), &resolved))?
                .unbind();
            Ok((resolved.unbind(), awaitable))
        })?;
        let result = task.settle(Ok(awaitable)).await?;
        Python::attach(|py| {
            let replacement = result?.into_bound(py).cast_into::<PyDict>()?;
            write_back(before.bind(py), &replacement, arguments.bind(py))?;
            Ok(arguments)
        })
    }

    async fn arguments_prepared(&mut self, _: &PythonTask, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| self.request = Some(arguments.clone_ref(py)));
        Ok(())
    }

    async fn transform_response(
        &mut self,
        task: &PythonTask,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<Py<PyAny>> {
        let awaitable = Python::attach(|py| -> PyResult<_> {
            let request = self.request.as_ref().ok_or_else(missing_state)?;
            let resolved = self.resolved(py, request.bind(py))?;
            Ok(Body::TransformResponse
                .call(py, (<&str>::from(self.operation), &response, resolved))?
                .unbind())
        })?;
        task.settle(Ok(awaitable)).await?
    }
}

impl PythonOwned for HandlerBody {
    fn close(&mut self, _: Python<'_>) {
        self.base = None;
        self.request = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.base)?;
        visit.call(&self.request)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use litellm_host::{hooks::CallHooks, lifecycle::Timing};
    use litellm_host_python::{PythonOwned, PythonTask};
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };
    use rstest::rstest;

    use super::HandlerBody;
    use crate::{
        LoggingOperation, PublicCall,
        test_support::{
            InitializedPython, drive, finish, initialized_python, local, local_dict, namespace, run,
        },
    };

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    fn body<'py>(py: Python<'py>, script: &CStr) -> (HandlerBody, Bound<'py, PyDict>) {
        let locals = namespace(py, script);
        let call = PublicCall::capture(
            &local_dict(&locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(&locals, "kwargs"),
        )
        .unwrap();
        let body = HandlerBody::new(py, LoggingOperation::Messages, &call, true).unwrap();
        (body, locals)
    }

    #[rstest]
    #[case::sync_messages(LoggingOperation::Messages, false, false)]
    #[case::async_messages(LoggingOperation::Messages, true, true)]
    #[case::async_ocr(LoggingOperation::Ocr, true, false)]
    #[case::async_completion(LoggingOperation::Completion, true, false)]
    #[case::async_responses(LoggingOperation::Responses, true, false)]
    fn only_the_async_messages_call_runs_body_hooks(
        #[case] operation: LoggingOperation,
        #[case] asynchronous: bool,
        #[case] composed: bool,
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let locals = namespace(py, c"bound = {}\nkwargs = {'logger': logger}");
            let call = PublicCall::capture(
                &local_dict(&locals, "bound"),
                &PyTuple::empty(py),
                &local_dict(&locals, "kwargs"),
            )
            .unwrap();
            assert_eq!(
                HandlerBody::new(py, operation, &call, asynchronous).is_some(),
                composed
            );
        });
    }

    /// The fake fan-out answers with the value the test's `logger.bodies` builds, and the
    /// caller's task resolves that "awaitable" to itself.
    fn echo(_: Python<'_>, _: usize, awaitable: &Py<PyAny>) -> PyResult<Py<PyAny>> {
        Python::attach(|py| Ok(awaitable.clone_ref(py)))
    }

    #[rstest]
    fn the_pre_request_fan_out_sees_the_resolved_call_and_only_its_changes_join_the_keywords(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(
                py,
                c"
opaque = object()
tools = [{'name': 'replacement'}]
bound = {'model': 'bound-model', 'timeout': 600, 'api_key': None, 'tools': None}
kwargs = {'model': 'kwargs-model', 'api_key': 'secret', 'opaque': opaque, 'logger': logger}
logger.bodies = {'prepare': lambda operation, request: {**request, 'tools': tools, 'temperature': 0.75}}
",
            );
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let driven = drive(
                py,
                &task,
                body.prepare_request(&task, kwargs.clone_ref(py)),
                echo,
            );
            let prepared = driven.outcome.unwrap();
            assert!(prepared.is(&kwargs));
            assert_eq!(driven.awaited.len(), 1);
            run(
                py,
                &locals,
                c"
[(name, (operation, request))] = logger.calls
assert name == 'prepare_body'
assert operation == 'messages'
assert request == {
    'model': 'kwargs-model', 'timeout': 600, 'api_key': 'secret', 'tools': None,
    'opaque': opaque, 'logger': logger,
}
assert kwargs['tools'] is tools
assert kwargs['temperature'] == 0.75
assert kwargs['opaque'] is opaque
assert set(kwargs) == {'model', 'api_key', 'opaque', 'logger', 'tools', 'temperature'}
assert bound == {'model': 'bound-model', 'timeout': 600, 'api_key': None, 'tools': None}
",
            );
        });
    }

    /// What the fan-out hands back is laid over the keyword layer key by key, and the only
    /// keys left alone are the ones still bound to the very object the resolved request had.
    #[rstest]
    #[case::new_key(
        "{**request, 'temperature': 0.75}",
        "{'logger', 'system', 'temperature'}",
        "kwargs['temperature'] == 0.75"
    )]
    #[case::changed_value(
        "{**request, 'system': 'rewritten'}",
        "{'logger', 'system'}",
        "kwargs['system'] == 'rewritten'"
    )]
    #[case::same_object_from_base(
        "{**request}",
        "{'logger', 'system'}",
        "'max_tokens' not in kwargs"
    )]
    #[case::equal_but_distinct_object(
        "{**request, 'stop': list(request['stop'])}",
        "{'logger', 'system', 'stop'}",
        "kwargs['stop'] == ['END'] and kwargs['stop'] is not bound['stop']"
    )]
    #[case::none_replaces_a_value(
        "{**request, 'system': None}",
        "{'logger', 'system'}",
        "kwargs['system'] is None"
    )]
    #[case::dropped_key_is_not_a_deletion(
        "{name: value for name, value in request.items() if name != 'system'}",
        "{'logger', 'system'}",
        "kwargs['system'] == 'caller'"
    )]
    fn only_the_keys_the_fan_out_changed_join_the_keyword_layer(
        #[case] replacement: &str,
        #[case] keys: &str,
        #[case] check: &str,
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let script = std::ffi::CString::new(format!(
                "bound = {{'model': 'bound-model', 'max_tokens': 8, 'stop': ['END']}}\nkwargs = {{'system': 'caller', 'logger': logger}}\nlogger.bodies = {{'prepare': lambda operation, request: {replacement}}}"
            ))
            .unwrap();
            let (mut body, locals) = body(py, &script);
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let prepared = drive(
                py,
                &task,
                body.prepare_request(&task, kwargs.clone_ref(py)),
                echo,
            )
            .outcome
            .unwrap();
            assert!(prepared.is(&kwargs));
            let assertion = std::ffi::CString::new(format!(
                "assert set(kwargs) == {keys}, set(kwargs)\nassert {check}, kwargs\nassert bound == {{'model': 'bound-model', 'max_tokens': 8, 'stop': ['END']}}"
            ))
            .unwrap();
            run(py, &locals, &assertion);
        });
    }

    #[rstest]
    #[case::exception("ValueError('pre-request rejected')")]
    #[case::cancellation("asyncio.CancelledError()")]
    fn a_failing_pre_request_fan_out_keeps_its_exception_and_leaves_the_keywords_alone(
        #[case] failure: &str,
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let script = std::ffi::CString::new(format!(
                "import asyncio\nbound = {{'model': 'bound-model'}}\nkwargs = {{'logger': logger}}\nfailure = {failure}"
            ))
            .unwrap();
            let (mut body, locals) = body(py, &script);
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let failure = local(&locals, "failure").unbind();
            let driven = drive(
                py,
                &task,
                body.prepare_request(&task, kwargs.clone_ref(py)),
                |py, _, _| Err(PyErr::from_value(failure.bind(py).clone())),
            );
            let error = driven.outcome.unwrap_err();
            assert!(error.value(py).is(failure.bind(py)));
            run(py, &locals, c"assert set(kwargs) == {'logger'}");
        });
    }

    #[rstest]
    fn the_response_fan_out_sees_the_response_and_the_prepared_request_and_its_answer_is_returned(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(
                py,
                c"
bound = {'model': 'bound-model', 'max_tokens': 8}
kwargs = {'model': 'kwargs-model', 'logger': logger}
response = {'content': [{'type': 'text', 'text': 'provider'}]}
replacement = {'content': [{'type': 'text', 'text': 'callback'}]}
logger.bodies = {'transform': lambda operation, response, request: replacement}
",
            );
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            finish(py, &task, body.arguments_prepared(&task, &kwargs)).unwrap();
            let response = local(&locals, "response").unbind();
            let replacement = local(&locals, "replacement").unbind();
            let driven = drive(
                py,
                &task,
                body.transform_response(&task, response.clone_ref(py), TIMING),
                echo,
            );
            let selected = driven.outcome.unwrap();
            assert!(selected.is(&replacement));
            run(
                py,
                &locals,
                c"
[(name, (operation, seen_response, request))] = logger.calls
assert name == 'transform_body'
assert operation == 'messages'
assert seen_response is response
assert request == {'model': 'kwargs-model', 'max_tokens': 8, 'logger': logger}
",
            );
        });
    }

    #[rstest]
    fn the_response_fan_out_needs_the_prepared_request(initialized_python: &InitializedPython) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(py, c"bound = {}\nkwargs = {'logger': logger}");
            let task = PythonTask::new(true);
            let response = py.None();
            let error =
                finish(py, &task, body.transform_response(&task, response, TIMING)).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            run(py, &locals, c"assert logger.calls == []");
        });
    }

    #[rstest]
    fn close_releases_the_retained_dicts_and_later_stages_report_missing_state(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(py, c"bound = {}\nkwargs = {'logger': logger}");
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            finish(py, &task, body.arguments_prepared(&task, &kwargs)).unwrap();
            body.close(py);
            body.close(py);
            let error =
                finish(py, &task, body.prepare_request(&task, kwargs.clone_ref(py))).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            run(py, &locals, c"assert logger.calls == []");
        });
    }
}
