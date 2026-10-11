//! The dispatched handler's body as the native call sees it. Python's handlers run body hooks
//! between the `@client` wrapper and the provider call: the async Messages handler runs
//! `async_pre_request_hook` over the resolved call before the provider transform and the
//! agentic-loop hooks over the complete response, and both Responses handlers run the
//! agentic-loop hooks over the complete response. The fan-outs stay in Python, keyed by
//! operation; an async call awaits them on the caller's task and a sync call runs them to
//! completion the way Python's sync handler does. This hook hands them the resolved request
//! and folds their answers back into the keyword view.

use litellm_host::{
    hooks::{CallHooks, StreamPolicy},
    lifecycle::Timing,
};
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
    prepares_request: bool,
    base: Option<Py<PyDict>>,
    request: Option<Py<PyDict>>,
    /// The caller asked for a stream and the request stage turned it off, so the complete
    /// response has to reach the caller as a stream.
    stream_turned_off: bool,
    /// The handler asked to hold the provider stream back for its response hooks.
    holds_back: bool,
}

fn streams(py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<bool> {
    match arguments.bind(py).get_item("stream")? {
        Some(value) => value.is_truthy(),
        None => Ok(false),
    }
}

/// Whether the operation's Python handler runs body hooks at all, and if so whether it runs
/// a request stage before the provider call.
fn body_stages(operation: LoggingOperation, asynchronous: bool) -> Option<bool> {
    match (operation, asynchronous) {
        (LoggingOperation::Messages, true) => Some(true),
        (LoggingOperation::Responses, _) => Some(false),
        (LoggingOperation::Messages, false)
        | (LoggingOperation::Completion, _)
        | (LoggingOperation::Ocr, _) => None,
    }
}

impl HandlerBody {
    /// The body hooks for a call, when Python's handler would run them: the async Messages
    /// route and both Responses routes. Every other call keeps its wrapper-only contract.
    pub fn new(
        py: Python<'_>,
        operation: LoggingOperation,
        call: &PublicCall,
        asynchronous: bool,
    ) -> Option<Self> {
        body_stages(operation, asynchronous).map(|prepares_request| Self {
            operation,
            prepares_request,
            base: Some(call.base(py)),
            request: None,
            stream_turned_off: false,
            holds_back: false,
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

/// A fan-out's answer: awaited on the caller's task for an async call, run to completion
/// for a sync one, as Python's sync handler runs the same coroutine.
async fn settle(task: &PythonTask, awaitable: Py<PyAny>) -> PyResult<Py<PyAny>> {
    if task.asynchronous() {
        return task.settle(Ok(awaitable)).await?;
    }
    Python::attach(|py| Body::RunSync.call(py, (awaitable,)).map(Bound::unbind))
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
        if !self.prepares_request {
            return Ok(arguments);
        }
        let (before, awaitable, streamed) = Python::attach(|py| -> PyResult<_> {
            let resolved = self.resolved(py, arguments.bind(py))?;
            let streamed = streams(py, &arguments)?;
            let awaitable = Body::PrepareRequest
                .call(py, (<&str>::from(self.operation), &resolved))?
                .unbind();
            Ok((resolved.unbind(), awaitable, streamed))
        })?;
        let replacement = settle(task, awaitable).await?;
        Python::attach(|py| {
            let replacement = replacement.into_bound(py).cast_into::<PyDict>()?;
            write_back(before.bind(py), &replacement, arguments.bind(py))?;
            self.stream_turned_off = streamed && !streams(py, &arguments)?;
            Ok(arguments)
        })
    }

    async fn stream_policy(
        &mut self,
        _: &PythonTask,
        arguments: &Py<PyDict>,
    ) -> PyResult<StreamPolicy> {
        if self.stream_turned_off {
            return Ok(StreamPolicy::Synthesize);
        }
        let holds_back = Python::attach(|py| -> PyResult<bool> {
            if !streams(py, arguments)? {
                return Ok(false);
            }
            let resolved = self.resolved(py, arguments.bind(py))?;
            Body::StreamPolicy
                .call(py, (<&str>::from(self.operation), resolved))?
                .extract::<String>()
                .map(|policy| policy == "hold_back")
        })?;
        self.holds_back = holds_back;
        Ok(if holds_back {
            StreamPolicy::HoldBack
        } else {
            StreamPolicy::Relay
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
                .call(
                    py,
                    (
                        <&str>::from(self.operation),
                        &response,
                        resolved,
                        self.holds_back,
                    ),
                )?
                .unbind())
        })?;
        settle(task, awaitable).await
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

    use litellm_host::{
        hooks::{CallHooks, StreamPolicy},
        lifecycle::Timing,
    };
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
        body_of(py, script, LoggingOperation::Messages, true)
    }

    fn body_of<'py>(
        py: Python<'py>,
        script: &CStr,
        operation: LoggingOperation,
        asynchronous: bool,
    ) -> (HandlerBody, Bound<'py, PyDict>) {
        let locals = namespace(py, script);
        let call = PublicCall::capture(
            &local_dict(&locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(&locals, "kwargs"),
        )
        .unwrap();
        let body = HandlerBody::new(py, operation, &call, asynchronous).unwrap();
        (body, locals)
    }

    #[rstest]
    #[case::sync_messages(LoggingOperation::Messages, false, false)]
    #[case::async_messages(LoggingOperation::Messages, true, true)]
    #[case::sync_responses(LoggingOperation::Responses, false, true)]
    #[case::async_responses(LoggingOperation::Responses, true, true)]
    #[case::async_ocr(LoggingOperation::Ocr, true, false)]
    #[case::sync_ocr(LoggingOperation::Ocr, false, false)]
    #[case::async_completion(LoggingOperation::Completion, true, false)]
    #[case::sync_completion(LoggingOperation::Completion, false, false)]
    fn the_calls_whose_python_handler_runs_body_hooks_compose_one(
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
logger.bodies = {'transform': lambda operation, response, request, stream: replacement}
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
[(name, (operation, seen_response, request, stream))] = logger.calls
assert name == 'transform_body'
assert operation == 'messages'
assert stream is False
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

    /// Responses has no pre-request fan-out: the request stage is a pass-through that never
    /// reaches Python, for the async and the sync route alike.
    #[rstest]
    #[case::asynchronous(true)]
    #[case::synchronous(false)]
    fn the_responses_body_leaves_the_request_stage_to_the_wrapper(
        #[case] asynchronous: bool,
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body_of(
                py,
                c"bound = {'model': 'bound-model'}\nkwargs = {'logger': logger}",
                LoggingOperation::Responses,
                asynchronous,
            );
            let task = PythonTask::new(asynchronous);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let driven = drive(
                py,
                &task,
                body.prepare_request(&task, kwargs.clone_ref(py)),
                echo,
            );
            assert!(driven.outcome.unwrap().is(&kwargs));
            assert!(driven.awaited.is_empty());
            run(py, &locals, c"assert logger.calls == []");
        });
    }

    /// A sync call cannot await the fan-out, so the body runs it to completion through the
    /// shim's runner and never suspends the caller's task.
    #[rstest]
    fn a_sync_responses_call_runs_the_response_fan_out_to_completion_without_suspending(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body_of(
                py,
                c"
bound = {'model': 'bound-model'}
kwargs = {'logger': logger}
response = {'output': []}
replacement = {'output': ['from the loop']}
logger.bodies = {'transform': lambda operation, response, request, stream: replacement}
",
                LoggingOperation::Responses,
                false,
            );
            let task = PythonTask::new(false);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            finish(py, &task, body.arguments_prepared(&task, &kwargs)).unwrap();
            let response = local(&locals, "response").unbind();
            let driven = drive(
                py,
                &task,
                body.transform_response(&task, response, TIMING),
                |_, _, _| panic!("a sync call never suspends"),
            );
            let selected = driven.outcome.unwrap();
            assert!(driven.awaited.is_empty());
            locals.set_item("selected", selected).unwrap();
            run(
                py,
                &locals,
                c"
assert selected is replacement
assert [name for name, _ in logger.calls] == ['transform_body', 'run_sync']
assert logger.calls[0][1][0] == 'responses'
",
            );
        });
    }

    /// The stream policy is the handler's call: a streaming request is held back exactly when
    /// the shim says so, a complete request is always relayed and never asks Python.
    #[rstest]
    #[case::complete_request(
        "kwargs = {'logger': logger}",
        "'hold_back'",
        StreamPolicy::Relay,
        false
    )]
    #[case::stream_relayed(
        "kwargs = {'stream': True, 'logger': logger}",
        "'relay'",
        StreamPolicy::Relay,
        true
    )]
    #[case::stream_held_back(
        "kwargs = {'stream': True, 'logger': logger}",
        "'hold_back'",
        StreamPolicy::HoldBack,
        true
    )]
    fn the_stream_policy_follows_the_handler(
        #[case] kwargs: &str,
        #[case] answer: &str,
        #[case] expected: StreamPolicy,
        #[case] asked_python: bool,
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let script = std::ffi::CString::new(format!(
                "bound = {{'model': 'bound-model', 'stream': False}}\n{kwargs}\nlogger.bodies = {{'policy': lambda operation, request: {answer}}}"
            ))
            .unwrap();
            let (mut body, locals) = body(py, &script);
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let policy = finish(py, &task, body.stream_policy(&task, &kwargs)).unwrap();
            assert_eq!(policy, expected);
            let calls = if asked_python { "['policy_body']" } else { "[]" };
            let asked = std::ffi::CString::new(format!("assert [name for name, _ in logger.calls] == {calls}")).unwrap();
            run(py, &locals, &asked);
            if asked_python {
                run(
                    py,
                    &locals,
                    c"assert logger.calls[0][1][0] == 'messages' and logger.calls[0][1][1]['stream'] is True",
                );
            }
        });
    }

    /// A pre-request fan-out that turns the stream off leaves a caller who asked for one, so
    /// the complete response has to be synthesized into a stream for them.
    #[rstest]
    fn a_pre_request_fan_out_that_turns_the_stream_off_makes_the_call_synthesize_one(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(
                py,
                c"
bound = {'model': 'bound-model', 'stream': False}
kwargs = {'stream': True, 'logger': logger}
logger.bodies = {'prepare': lambda operation, request: {**request, 'stream': False}}
",
            );
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            drive(
                py,
                &task,
                body.prepare_request(&task, kwargs.clone_ref(py)),
                echo,
            )
            .outcome
            .unwrap();
            let policy = finish(py, &task, body.stream_policy(&task, &kwargs)).unwrap();
            assert_eq!(policy, StreamPolicy::Synthesize);
            run(py, &locals, c"assert kwargs['stream'] is False");
        });
    }

    #[rstest]
    fn the_response_fan_out_learns_whether_the_stream_was_held_back(
        initialized_python: &InitializedPython,
    ) {
        initialized_python.attach(|py| {
            let (mut body, locals) = body(
                py,
                c"
bound = {'model': 'bound-model'}
kwargs = {'stream': True, 'logger': logger}
response = {'content': []}
logger.bodies = {
    'policy': lambda operation, request: 'hold_back',
    'transform': lambda operation, response, request, stream: response,
}
",
            );
            let task = PythonTask::new(true);
            let kwargs = local_dict(&locals, "kwargs").unbind();
            finish(py, &task, body.stream_policy(&task, &kwargs)).unwrap();
            finish(py, &task, body.arguments_prepared(&task, &kwargs)).unwrap();
            let response = local(&locals, "response").unbind();
            drive(
                py,
                &task,
                body.transform_response(&task, response, TIMING),
                echo,
            )
            .outcome
            .unwrap();
            run(
                py,
                &locals,
                c"assert logger.calls[-1][0] == 'transform_body' and logger.calls[-1][1][3] is True",
            );
        });
    }
}
