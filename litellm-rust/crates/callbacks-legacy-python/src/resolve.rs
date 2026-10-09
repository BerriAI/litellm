use litellm_host::{call::Operation, hooks::CallHooks, lifecycle::Timing};
use litellm_host_python::{HookStep, PythonOwned, PythonRuntime, missing_state};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use crate::{PublicCall, call::resolved, python::Body};

pub(crate) struct Resolve {
    base: Option<Py<PyDict>>,
    body: bool,
    request: Option<Py<PyDict>>,
}

type Step<T> = PyResult<HookStep<Resolve, T>>;

fn runs_body_hooks(operation: Operation, asynchronous: bool) -> bool {
    matches!((operation, asynchronous), (Operation::Messages, true))
}

impl Resolve {
    pub(crate) fn new(
        py: Python<'_>,
        operation: Operation,
        call: &PublicCall,
        asynchronous: bool,
    ) -> Self {
        Self {
            base: Some(call.base(py)),
            body: runs_body_hooks(operation, asynchronous),
            request: None,
        }
    }

    fn resolved<'py>(&self, py: Python<'py>, kwargs: &Py<PyDict>) -> PyResult<Bound<'py, PyDict>> {
        let base = self.base.as_ref().ok_or_else(missing_state)?;
        resolved(base.bind(py), kwargs.bind(py))
    }
}

impl CallHooks<PythonRuntime> for Resolve {
    fn prepare_request(&mut self, py: Python<'_>, kwargs: Py<PyDict>) -> Step<Py<PyDict>> {
        let resolved = self.resolved(py, &kwargs)?;
        if !self.body {
            return Ok(HookStep::Ready(resolved.unbind()));
        }
        let before = resolved.clone().unbind();
        let awaitable = Body::PrepareRequest.call(py, (resolved,))?;
        Ok(HookStep::Await(
            awaitable.unbind(),
            Box::new(move |_, py, result| {
                let replacement = result?.into_bound(py).cast_into::<PyDict>()?;
                let before = before.bind(py);
                let layer = kwargs.bind(py);
                for (name, value) in replacement.iter() {
                    let unchanged = before
                        .get_item(&name)?
                        .is_some_and(|previous| previous.is(&value));
                    if !unchanged {
                        layer.set_item(name, value)?;
                    }
                }
                Ok(HookStep::Ready(replacement.unbind()))
            }),
        ))
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.request = Some(arguments.clone_ref(py));
        Ok(())
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> Step<Py<PyAny>> {
        if !self.body {
            return Ok(HookStep::Ready(response));
        }
        let request = self.request.as_ref().ok_or_else(missing_state)?;
        let awaitable = Body::TransformResponse.call(py, (&response, request))?;
        Ok(HookStep::Await(
            awaitable.unbind(),
            Box::new(|_, _, result| result.map(HookStep::Ready)),
        ))
    }
}

impl PythonOwned for Resolve {
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

    use litellm_host::call::Operation;
    use litellm_host::hooks::CallHooks;
    use litellm_host::lifecycle::Timing;
    use litellm_host_python::{HookStep, PythonOwned};
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };
    use rstest::rstest;

    use super::Resolve;
    use crate::PublicCall;
    use crate::test_support::{local, local_dict, namespace, run};

    const TIMING: Timing = Timing {
        start_time: 0.0,
        end_time: 1.0,
    };

    /// The namespace's `body` answers both fan-outs; it travels in `bound` so every
    /// resolved request carries it.
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

    fn resolve<'py>(
        py: Python<'py>,
        script: &CStr,
        operation: Operation,
        asynchronous: bool,
    ) -> (Resolve, Bound<'py, PyDict>) {
        let locals = namespace(py, BODY);
        run(py, &locals, script);
        let call = PublicCall::capture(
            &local_dict(&locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(&locals, "kwargs"),
        )
        .unwrap();
        (Resolve::new(py, operation, &call, asynchronous), locals)
    }

    fn finish<T>(py: Python<'_>, hooks: &mut Resolve, step: HookStep<Resolve, T>) -> PyResult<T> {
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
    #[case::sync_messages(Operation::Messages, false)]
    #[case::async_ocr(Operation::Ocr, true)]
    #[case::async_completion(Operation::Completion, true)]
    #[case::async_responses(Operation::Responses, true)]
    fn every_other_call_resolves_the_keywords_over_the_base_without_python(
        #[case] operation: Operation,
        #[case] asynchronous: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = resolve(
                py,
                c"
bound = {'model': 'bound-model', 'timeout': 600, 'api_key': None}
kwargs = {'model': 'kwargs-model', 'api_key': 'secret'}
",
                operation,
                asynchronous,
            );
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let step = hooks.prepare_request(py, kwargs.clone_ref(py)).unwrap();
            assert!(matches!(step, HookStep::Ready(_)));
            let resolved = finish(py, &mut hooks, step).unwrap();
            locals.set_item("resolved", &resolved).unwrap();
            run(
                py,
                &locals,
                c"
assert resolved == {'model': 'kwargs-model', 'timeout': 600, 'api_key': 'secret'}, resolved
assert resolved is not kwargs and resolved is not bound
assert calls == []
assert kwargs == {'model': 'kwargs-model', 'api_key': 'secret'}
",
            );
            hooks.arguments_prepared(py, &resolved).unwrap();
            let step = hooks
                .transform_response(py, local(&locals, "bound").unbind(), TIMING)
                .unwrap();
            let HookStep::Ready(response) = step else {
                panic!("no body hook runs for this call");
            };
            assert!(response.bind(py).is(local(&locals, "bound")));
        });
    }

    #[rstest]
    fn the_pre_request_fan_out_sees_the_resolved_call_and_only_its_rewrites_join_the_keywords() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = resolve(
                py,
                c"
tools = [{'name': 'lookup'}]
def prepared(request):
    assert request == {**bound, **kwargs}
    return {**request, 'tools': tools}
bound = {'model': 'anthropic/claude', 'stream': False, 'tool_choice': None, 'body': Body(prepared=prepared)}
kwargs = {'model': 'anthropic/claude', 'temperature': 0.25}
",
                Operation::Messages,
                true,
            );
            let kwargs = local_dict(&locals, "kwargs").unbind();
            let step = hooks.prepare_request(py, kwargs.clone_ref(py)).unwrap();
            let prepared = finish(py, &mut hooks, step).unwrap();
            locals.set_item("prepared", prepared).unwrap();
            run(
                py,
                &locals,
                c"
assert prepared['tools'] is tools
assert prepared['temperature'] == 0.25
assert prepared['stream'] is False
assert kwargs == {'model': 'anthropic/claude', 'temperature': 0.25, 'tools': tools}, kwargs
assert kwargs['tools'] is tools
assert 'tools' not in bound
",
            );
        });
    }

    #[rstest]
    fn the_agentic_loop_sees_the_prepared_request_and_replaces_the_response() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = resolve(
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
prepared = {'model': 'anthropic/claude', 'max_tokens': 8, 'body': body, 'litellm_logging_obj': logger}
",
                Operation::Messages,
                true,
            );
            let prepared = local_dict(&locals, "prepared").unbind();
            hooks.arguments_prepared(py, &prepared).unwrap();
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
assert request is prepared
",
            );
        });
    }

    #[rstest]
    fn a_failed_fan_out_keeps_its_exception_and_a_closed_body_cannot_run() {
        Python::initialize();
        Python::attach(|py| {
            let (mut hooks, locals) = resolve(
                py,
                c"
failure = RuntimeError('rejected')
def prepared(request):
    raise failure
bound = {'body': Body(prepared=prepared)}
kwargs = {}
",
                Operation::Messages,
                true,
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
}
