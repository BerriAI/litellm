//! The Messages body as the Python handler runs it around the provider call: the
//! pre-request callback fan-out before the request is decoded, and the agentic loop over
//! the provider's response. Both see the call as the handler does, the keyword dict laid
//! over the signature base, and the pre-request replacement becomes the call's keyword
//! layer so every later reader sees it.

use litellm_host::{hooks::CallHooks, lifecycle::Timing};
use litellm_host_python::{HookStep, PythonOwned, PythonRuntime, effective_py_args, missing_state};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{
    CallBoundary, CallbackMapping, LoggingOperation, PublicCall, mapping::Binding, python::Body,
};

pub struct MessagesBody {
    base: Option<Py<PyDict>>,
    request: Option<Py<PyDict>>,
}

type Step<T> = PyResult<HookStep<MessagesBody, T>>;
type Prepare = fn(&mut MessagesBody, Python<'_>, Py<PyDict>) -> Step<Py<PyDict>>;
type Transform = fn(&mut MessagesBody, Python<'_>, Py<PyAny>, Timing) -> Step<Py<PyAny>>;

const PREPARE: Binding<Prepare> = Binding {
    boundary: CallBoundary::PrepareRequest,
    invoke: MessagesBody::run_pre_request_hooks,
    callbacks: &["async_pre_request_hook"],
};

const TRANSFORM: Binding<Transform> = Binding {
    boundary: CallBoundary::TransformResponse,
    invoke: MessagesBody::run_agentic_loop,
    callbacks: &[
        "async_should_run_agentic_loop",
        "async_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_post_agentic_loop_response_hook",
    ],
};

pub(crate) fn callback_mappings() -> impl Iterator<Item = CallbackMapping> {
    PREPARE.mappings().chain(TRANSFORM.mappings())
}

/// The body hooks a native call runs, when the Python handler would run any. The sync
/// Messages handler runs neither fan-out, so neither does the native sync call.
pub fn body_hooks(
    py: Python<'_>,
    operation: LoggingOperation,
    call: &PublicCall,
    asynchronous: bool,
) -> Option<MessagesBody> {
    match (operation, asynchronous) {
        (LoggingOperation::Messages, true) => Some(MessagesBody::new(py, call)),
        _ => None,
    }
}

impl MessagesBody {
    pub(crate) fn new(py: Python<'_>, call: &PublicCall) -> Self {
        Self {
            base: Some(call.base(py)),
            request: None,
        }
    }

    fn effective<'py>(&self, py: Python<'py>, kwargs: &Py<PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let base = self.base.as_ref().ok_or_else(missing_state)?;
        effective_py_args(base.bind(py), kwargs.bind(py))
    }

    fn run_pre_request_hooks(&mut self, py: Python<'_>, arguments: Py<PyDict>) -> Step<Py<PyDict>> {
        let awaitable = Body::PrepareRequest.call(py, (self.effective(py, &arguments)?,))?;
        Ok(HookStep::Await(awaitable.unbind(), Self::resume_request))
    }

    fn resume_request(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Step<Py<PyDict>> {
        Ok(HookStep::Ready(
            result?.into_bound(py).cast_into::<PyDict>()?.unbind(),
        ))
    }

    fn run_agentic_loop(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> Step<Py<PyAny>> {
        let request = self.request.as_ref().ok_or_else(missing_state)?;
        let awaitable =
            Body::TransformResponse.call(py, (&response, self.effective(py, request)?))?;
        Ok(HookStep::Await(awaitable.unbind(), Self::resume_response))
    }

    fn resume_response(&mut self, _: Python<'_>, result: PyResult<Py<PyAny>>) -> Step<Py<PyAny>> {
        result.map(HookStep::Ready)
    }
}

impl CallHooks<PythonRuntime> for MessagesBody {
    fn prepare_request(&mut self, py: Python<'_>, arguments: Py<PyDict>) -> Step<Py<PyDict>> {
        (PREPARE.invoke)(self, py, arguments)
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.request = Some(arguments.clone_ref(py));
        Ok(())
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> Step<Py<PyAny>> {
        (TRANSFORM.invoke)(self, py, response, timing)
    }
}

impl PythonOwned for MessagesBody {
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

    use litellm_host::hooks::CallHooks;
    use litellm_host::lifecycle::Timing;
    use litellm_host_python::{HookStep, PythonOwned};
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };
    use rstest::rstest;

    use super::{MessagesBody, body_hooks};
    use crate::test_support::{local, local_dict, namespace, run};
    use crate::{LoggingOperation, PublicCall};

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    /// The namespace's `body` answers both fan-outs; it travels in `bound` so every
    /// effective request carries it.
    const BODY: &CStr = c"
import asyncio
calls = []
class Body:
    def __init__(self, prepared=None, transformed=None):
        self.prepared = prepared
        self.transformed = transformed
    async def prepare(self, request):
        await asyncio.sleep(0)
        calls.append(('prepare', request))
        return self.prepared(request)
    async def transform(self, response, request):
        await asyncio.sleep(0)
        calls.append(('transform', response, request))
        return self.transformed(response)
";

    fn body<'py>(py: Python<'py>, script: &CStr) -> (MessagesBody, Bound<'py, PyDict>) {
        let locals = namespace(py, BODY);
        run(py, &locals, script);
        let call = PublicCall::capture(
            &local_dict(&locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(&locals, "kwargs"),
        )
        .unwrap();
        (MessagesBody::new(py, &call), locals)
    }

    fn finish<T>(
        py: Python<'_>,
        hooks: &mut MessagesBody,
        step: HookStep<MessagesBody, T>,
    ) -> PyResult<T> {
        match step {
            HookStep::Ready(value) => Ok(value),
            HookStep::Await(awaitable, resume) => {
                let value = py
                    .import("asyncio")?
                    .call_method1("run", (awaitable,))
                    .map(Bound::unbind);
                let next = resume(hooks, py, value)?;
                finish(py, hooks, next)
            }
        }
    }

    #[rstest]
    fn the_pre_request_fan_out_sees_the_resolved_call_and_its_answer_becomes_the_keywords() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = body(
                py,
                c"
tools = [{'name': 'lookup'}]
def prepared(request):
    assert request == {**bound, **kwargs}
    return {**request, 'tools': tools}
bound = {'model': 'anthropic/claude', 'stream': False, 'tool_choice': None, 'body': Body(prepared=prepared)}
kwargs = {'model': 'anthropic/claude', 'temperature': 0.25}
",
            );
            let step = hooks
                .prepare_request(py, local_dict(&locals, "kwargs").unbind())
                .unwrap();
            let prepared = finish(py, &mut hooks, step).unwrap();
            locals.set_item("prepared", prepared).unwrap();
            run(
                py,
                &locals,
                c"
assert prepared['tools'] is tools
assert prepared['temperature'] == 0.25
assert prepared['stream'] is False
assert 'tools' not in kwargs
",
            );
        });
    }

    #[rstest]
    fn the_agentic_loop_sees_the_adopted_request_and_replaces_the_response() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = body(
                py,
                c"
original = object()
replacement = object()
def transformed(response):
    assert response is original
    return replacement
body = Body(transformed=transformed)
bound = {'model': 'anthropic/claude', 'max_tokens': 8, 'body': body}
kwargs = {'model': 'anthropic/claude'}
adopted = {'model': 'anthropic/claude', 'litellm_logging_obj': logger}
",
            );
            let adopted = local_dict(&locals, "adopted").unbind();
            hooks.arguments_prepared(py, &adopted).unwrap();
            let step = hooks
                .transform_response(py, local(&locals, "original").unbind(), TIMING)
                .unwrap();
            let transformed = finish(py, &mut hooks, step).unwrap();
            locals.set_item("transformed", transformed).unwrap();
            run(
                py,
                &locals,
                c"
assert transformed is replacement
kind, response, request = calls[-1]
assert request == {'model': 'anthropic/claude', 'max_tokens': 8, 'body': body, 'litellm_logging_obj': logger}
assert request['litellm_logging_obj'] is logger
",
            );
        });
    }

    #[rstest]
    fn a_failed_fan_out_keeps_its_exception_and_a_closed_body_cannot_run() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = body(
                py,
                c"
failure = RuntimeError('rejected')
def prepared(request):
    raise failure
bound = {'body': Body(prepared=prepared)}
kwargs = {}
",
            );
            let step = hooks.prepare_request(py, PyDict::new(py).unbind()).unwrap();
            let error = finish(py, &mut hooks, step).unwrap_err();
            assert!(error.value(py).is(local(&locals, "failure")));
            assert!(hooks.transform_response(py, py.None(), TIMING).is_err());
            hooks.close(py);
            hooks.close(py);
            assert!(hooks.prepare_request(py, PyDict::new(py).unbind()).is_err());
        });
    }

    #[rstest]
    #[case::async_messages(LoggingOperation::Messages, true, true)]
    #[case::sync_messages(LoggingOperation::Messages, false, false)]
    #[case::async_ocr(LoggingOperation::Ocr, true, false)]
    #[case::async_completion(LoggingOperation::Completion, true, false)]
    #[case::async_responses(LoggingOperation::Responses, true, false)]
    fn only_the_asynchronous_messages_call_runs_body_hooks(
        #[case] operation: LoggingOperation,
        #[case] asynchronous: bool,
        #[case] expected: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let call = PublicCall::capture(&PyDict::new(py), &PyTuple::empty(py), &PyDict::new(py))
                .unwrap();
            assert_eq!(body_hooks(py, operation, &call, asynchronous).is_some(), expected);
        });
    }
}
