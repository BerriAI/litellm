use std::{future::Future, pin::pin, sync::Mutex};

use futures_util::future::BoxFuture;
use litellm_host::{
    hooks::CallHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};
use litellm_host_python::{
    HookChain, PythonCallEvent, PythonOwned, PythonRuntime, PythonTask, Resumed, Suspension,
};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};
use rstest::{fixture, rstest};

struct ScriptHooks {
    object: Py<PyAny>,
}

impl ScriptHooks {
    fn invoke(
        &self,
        py: Python<'_>,
        task: &PythonTask,
        name: &str,
        value: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        if task.asynchronous() {
            self.object.call_method1(py, "invoke", (name, value))
        } else {
            self.object.call_method1(py, name, (value,))
        }
    }

    async fn call(
        &self,
        task: &PythonTask,
        name: &str,
        value: Py<PyAny>,
    ) -> PyResult<PyResult<Py<PyAny>>> {
        let result = Python::attach(|py| self.invoke(py, task, name, value));
        task.settle(result).await
    }
}

impl PythonOwned for ScriptHooks {
    fn close(&mut self, py: Python<'_>) {
        self.object = py.None();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.object)
    }
}

impl CallHooks<PythonRuntime> for ScriptHooks {
    async fn prepare_arguments(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
        _: f64,
    ) -> PyResult<Py<PyDict>> {
        let result = self.call(task, "prepare", arguments.into_any()).await?;
        Python::attach(|py| Ok(result?.into_bound(py).cast_into::<PyDict>()?.unbind()))
    }

    async fn prepare_request(
        &mut self,
        task: &PythonTask,
        arguments: Py<PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let result = self.call(task, "request", arguments.into_any()).await?;
        Python::attach(|py| Ok(result?.into_bound(py).cast_into::<PyDict>()?.unbind()))
    }

    async fn arguments_prepared(&mut self, _: &PythonTask, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "adopt", (arguments,)))?;
        Ok(())
    }

    async fn before_provider_request(
        &mut self,
        task: &PythonTask,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        let url = Python::attach(|py| {
            self.object.bind(py).setattr("model", &context.model)?;
            Ok::<_, PyErr>(wire.url.clone().into_pyobject(py)?.into_any().unbind())
        })?;
        let result = self.call(task, "before", url).await?;
        let url = Python::attach(|py| result?.extract::<String>(py))?;
        Ok(Box::new(WireRequest { url, ..*wire }))
    }

    async fn transform_response(
        &mut self,
        task: &PythonTask,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<Py<PyAny>> {
        self.call(task, "transform", response).await?
    }

    async fn on_event(&mut self, task: &PythonTask, event: PythonCallEvent<'_>) -> PyResult<()> {
        let value = Python::attach(|py| {
            let (name, value) = match event {
                PythonCallEvent::Succeeded { response, .. } => ("success", response.clone_ref(py)),
                PythonCallEvent::Failed { error, .. } => {
                    ("failure", error.clone_ref(py).into_value(py).into_any())
                }
                PythonCallEvent::Started { .. } => ("started", py.None()),
                PythonCallEvent::Cancelled { .. } => ("cancelled", py.None()),
                PythonCallEvent::Execution(_) => ("provider", py.None()),
            };
            Ok::<_, PyErr>(
                PyTuple::new(py, [name.into_pyobject(py)?.into_any().unbind(), value])?
                    .into_any()
                    .unbind(),
            )
        })?;
        self.call(task, "event", value).await?.map(|_| ())
    }

    async fn on_stream_open(&mut self, _: &PythonTask, _head: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "stream", (py.None(),)))?;
        Ok(())
    }

    async fn on_stream_chunk(&mut self, _: &PythonTask, chunk: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "stream", (chunk,)))?;
        Ok(())
    }
}

#[fixture]
fn scripts() -> Py<PyDict> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        py.run(
            c"
import asyncio
log = []
class Hooks:
    def __init__(self, name, delegate=None):
        self.name = name
        self.delegate = delegate
        self.error = None
        self.adopted = None
    async def invoke(self, name, value):
        if self.delegate is not None:
            return await self.delegate(name, value)
        await asyncio.sleep(0)
        return getattr(self, name)(value)
    def prepare(self, value):
        log.append((self.name, 'prepare', value))
        if self.error:
            raise self.error
        return {**value, 'order': value.get('order', '') + self.name}
    def request(self, value):
        log.append((self.name, 'request', value))
        return {**value, 'request': value.get('request', '') + self.name}
    def adopt(self, value):
        self.adopted = value
    def before(self, value):
        return value + self.name
    def transform(self, value):
        return (value, self.name)
    def event(self, value):
        log.append((self.name, *value))
        if self.error:
            raise self.error
    def stream(self, value):
        log.append((self.name, 'stream', value))
first = Hooks('a')
second = Hooks('b')
third = Hooks('c')
fourth = Hooks('d')
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        locals.unbind()
    })
}

fn hook(py: Python<'_>, scripts: &Py<PyDict>, name: &str) -> ScriptHooks {
    ScriptHooks {
        object: scripts.bind(py).get_item(name).unwrap().unwrap().unbind(),
    }
}

fn chain(py: Python<'_>, scripts: &Py<PyDict>) -> HookChain {
    HookChain::new()
        .with(hook(py, scripts, "first"))
        .with(hook(py, scripts, "second"))
}

/// Drives a hook to completion on the test thread, running each awaitable it suspends on
/// to completion with `asyncio.run`.
fn finish<T>(py: Python<'_>, task: &PythonTask, future: impl Future<Output = T>) -> T {
    let mut future = pin!(future);
    let mut answer = None;
    loop {
        match task.resume(future.as_mut(), answer.take()).unwrap() {
            Resumed::Complete(value) => return value,
            Resumed::Suspended(Suspension::Await(awaitable)) => {
                answer = Some(
                    py.import("asyncio")
                        .unwrap()
                        .call_method1("run", (awaitable,))
                        .map(Bound::unbind),
                );
            }
            Resumed::Suspended(_) => panic!("hooks never reach a stream boundary"),
        }
    }
}

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

struct PreparedPolicy;

impl CallHooks<PythonRuntime> for PreparedPolicy {
    async fn arguments_prepared(&mut self, _: &PythonTask, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| arguments.bind(py).set_item("policy", "configured"))
    }
}

impl PythonOwned for PreparedPolicy {
    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

#[rstest]
#[case::sync(false)]
#[case::suspended(true)]
fn transformations_feed_each_other_and_notifications_share_final_values(
    scripts: Py<PyDict>,
    #[case] asynchronous: bool,
) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = chain(py, &scripts).with(PreparedPolicy);
        let original = PyDict::new(py).unbind();
        let prepared = finish(
            py,
            &task,
            hooks.prepare_arguments(&task, original.clone_ref(py), 0.0),
        )
        .unwrap();
        let arguments = finish(py, &task, hooks.prepare_request(&task, prepared)).unwrap();
        finish(py, &task, hooks.arguments_prepared(&task, &arguments)).unwrap();
        let context = RequestContext {
            model: "model".into(),
            custom_llm_provider: "provider".into(),
            optional_params: serde_json::Value::Null,
            secret_fields: vec![],
            api_key: None,
        };
        let wire = Box::new(WireRequest {
            url: "url".into(),
            headers: vec![],
            body: serde_json::Value::Null,
        });
        let rewritten = finish(
            py,
            &task,
            hooks.before_provider_request(&task, wire, &context),
        )
        .unwrap();
        assert_eq!(rewritten.url, "urlab");
        let response = PyDict::new(py).unbind().into_any();
        let final_response = finish(
            py,
            &task,
            hooks.transform_response(&task, response.clone_ref(py), TIMING),
        )
        .unwrap();
        finish(
            py,
            &task,
            hooks.on_event(
                &task,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &final_response,
                },
            ),
        )
        .unwrap();
        finish(py, &task, hooks.on_stream_open(&task, &py.None())).unwrap();
        finish(py, &task, hooks.on_stream_chunk(&task, &response)).unwrap();
        let locals = scripts.bind(py);
        locals.set_item("arguments", arguments).unwrap();
        locals.set_item("original", original).unwrap();
        locals.set_item("response", response).unwrap();
        locals.set_item("final_response", final_response).unwrap();
        py.run(
            c"
assert arguments['order'] == 'ab'
assert arguments['request'] == 'ab'
assert original == {}
assert first.adopted is arguments and second.adopted is arguments
assert first.adopted['policy'] == second.adopted['policy'] == 'configured'
assert first.model == second.model == 'model'
assert final_response == ((response, 'b'), 'a')
assert [(name, kind) for name, kind, value in log] == [
    ('a', 'prepare'), ('b', 'prepare'), ('a', 'request'), ('b', 'request'),
    ('a', 'success'), ('b', 'success'),
    ('a', 'stream'), ('b', 'stream'), ('a', 'stream'), ('b', 'stream'),
]
assert log[4][2] is final_response and log[5][2] is final_response
assert log[8][2] is response and log[9][2] is response
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[rstest]
#[case::sync_success(false, false)]
#[case::async_success(true, false)]
#[case::sync_failure(false, true)]
#[case::async_failure(true, true)]
fn terminal_failure_does_not_skip_later_hooks(
    scripts: Py<PyDict>,
    #[case] asynchronous: bool,
    #[case] failed: bool,
    #[values("first", "second")] failing_hook: &str,
) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = chain(py, &scripts).with(hook(py, &scripts, "third"));
        let locals = scripts.bind(py);
        locals
            .get_item(failing_hook)
            .unwrap()
            .unwrap()
            .setattr(
                "error",
                pyo3::exceptions::PyRuntimeError::new_err("callback failed").into_value(py),
            )
            .unwrap();
        let error = pyo3::exceptions::PyValueError::new_err("provider failed");
        let response = py.None();
        let event = if failed {
            PythonCallEvent::Failed {
                timing: TIMING,
                origin: FailureOrigin::Call,
                error: &error,
            }
        } else {
            PythonCallEvent::Succeeded {
                timing: TIMING,
                response: &response,
            }
        };
        finish(py, &task, hooks.on_event(&task, event)).unwrap();
        locals
            .set_item(
                "selected",
                if failed {
                    error.into_value(py).into_any()
                } else {
                    response
                },
            )
            .unwrap();
        py.run(
            c"
assert [name for name, kind, value in log] == ['a', 'b', 'c']
assert all(kind == log[0][1] for name, kind, value in log)
assert all(value is selected for name, kind, value in log)
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[rstest]
#[case::sync(false)]
#[case::async_(true)]
fn transformation_failure_stops_the_chain(scripts: Py<PyDict>, #[case] asynchronous: bool) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = chain(py, &scripts);
        let locals = scripts.bind(py);
        py.run(
            c"first.error = ValueError('prepare failed')",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let result = finish(
            py,
            &task,
            hooks.prepare_arguments(&task, PyDict::new(py).unbind(), 0.0),
        );
        assert!(
            result.unwrap_err().value(py).is(locals
                .get_item("first")
                .unwrap()
                .unwrap()
                .getattr("error")
                .unwrap())
        );
        py.run(
            c"assert len(log) == 1 and log[0][0] == 'a'",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[rstest]
#[case::sync(false)]
#[case::async_(true)]
fn cancellation_stops_notification_dispatch(scripts: Py<PyDict>, #[case] asynchronous: bool) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = chain(py, &scripts);
        let locals = scripts.bind(py);
        py.run(
            c"first.error = asyncio.CancelledError()",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let response = py.None();
        let result = finish(
            py,
            &task,
            hooks.on_event(
                &task,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &response,
                },
            ),
        );
        assert!(
            result
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py)
        );
        py.run(
            c"assert len(log) == 1 and log[0][0] == 'a'",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

/// A preparation run as the caller's coroutine: the body owns the chain and awaits it on
/// the caller's task through the execution protocol.
struct Preparing {
    task: PythonTask,
    body: Mutex<Option<BoxFuture<'static, PyResult<Py<PyAny>>>>>,
}

impl Preparing {
    fn new(task: PythonTask, mut hooks: HookChain, arguments: Py<PyDict>) -> Self {
        let body_task = task.clone();
        let body = async move {
            let prepared = hooks.prepare_arguments(&body_task, arguments, 0.0).await?;
            hooks.arguments_prepared(&body_task, &prepared).await?;
            Ok(prepared.into_any())
        };
        Self {
            task,
            body: Mutex::new(Some(Box::pin(body))),
        }
    }
}

impl litellm_host_python::ExecutionBody for Preparing {
    fn resume(
        &mut self,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<litellm_host_python::ExecutionStep> {
        let mut body = self.body.lock().unwrap();
        let future = body.as_mut().expect("the preparation already completed");
        match self.task.resume(future.as_mut(), result)? {
            Resumed::Suspended(suspension) => {
                Ok(litellm_host_python::ExecutionStep::Suspend(suspension))
            }
            Resumed::Complete(outcome) => {
                *body = None;
                outcome.map(litellm_host_python::ExecutionStep::Return)
            }
        }
    }

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

fn lifecycle(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let source =
        std::ffi::CString::new(include_str!("../../../../litellm/rust_bridge/lifecycle.py"))
            .unwrap();
    PyModule::from_code(py, &source, c"lifecycle.py", c"hook_chain_test_lifecycle")
}

/// Every hook of one call runs in the caller's task and context: a `contextvars` write in
/// one hook is read by the next and by the caller after the call.
#[rstest]
fn suspended_hooks_keep_the_callers_task_and_context(scripts: Py<PyDict>) {
    Python::attach(|py| {
        let locals = scripts.bind(py);
        py.run(
            c"
import contextvars
import threading
state = contextvars.ContextVar('state')
async def invoke(name, value):
    assert asyncio.current_task() is caller
    assert threading.get_ident() == thread
    if name == 'prepare':
        state.set(state.get() + 'x')
    await asyncio.sleep(0)
    assert asyncio.current_task() is caller
    return {**value, 'context': state.get()}
first = Hooks('a', invoke)
second = Hooks('b', invoke)
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let task = PythonTask::new(true);
        let execution = litellm_host_python::Execution::new(
            Preparing::new(task, chain(py, &scripts), PyDict::new(py).unbind()),
            lifecycle,
        );
        locals
            .set_item("call", execution.into_coroutine(py).unwrap())
            .unwrap();
        py.run(
            c"
async def exercise():
    global caller, thread
    caller = asyncio.current_task()
    thread = threading.get_ident()
    state.set('caller')
    result = await call
    assert result['context'] == 'callerxx'
    assert state.get() == 'callerxx'
    assert first.adopted is result and second.adopted is result
asyncio.run(exercise())
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[pyclass(weakref)]
struct HookOwner {
    hooks: HookChain,
}

#[pymethods]
impl HookOwner {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.hooks.traverse(&visit)
    }

    fn __clear__(&mut self, py: Python<'_>) {
        self.hooks.close(py);
    }
}

/// A notification suspended on a hook's awaitable keeps its payload only as long as the
/// suspended future lives: dropping it leaves the owner and payload collectable.
#[rstest]
#[case::response(false)]
#[case::exception(true)]
fn a_dropped_suspended_notification_releases_its_payload(
    scripts: Py<PyDict>,
    #[case] failed: bool,
) {
    Python::attach(|py| {
        let task = PythonTask::new(true);
        let owner = Py::new(
            py,
            HookOwner {
                hooks: chain(py, &scripts),
            },
        )
        .unwrap();
        let locals = scripts.bind(py);
        locals.set_item("owner", &owner).unwrap();
        py.run(
            c"
import gc
import weakref
class Payload(Exception):
    pass
payload = Payload()
payload.owner = owner
owner_ref = weakref.ref(owner)
payload_ref = weakref.ref(payload)
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let payload = locals.get_item("payload").unwrap().unwrap().unbind();
        let error = PyErr::from_value(payload.bind(py).clone());
        {
            let mut hooks = owner.borrow_mut(py);
            let event = if failed {
                PythonCallEvent::Failed {
                    timing: TIMING,
                    origin: FailureOrigin::Call,
                    error: &error,
                }
            } else {
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &payload,
                }
            };
            let mut notification = pin!(hooks.hooks.on_event(&task, event));
            let Resumed::Suspended(Suspension::Await(awaitable)) =
                task.resume(notification.as_mut(), None).unwrap()
            else {
                panic!("notification must suspend")
            };
            awaitable.call_method0(py, "close").unwrap();
        }
        drop(error);
        drop(payload);
        drop(owner);
        py.run(
            c"
log.clear()
del owner, payload
gc.collect()
assert owner_ref() is None
assert payload_ref() is None
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[rstest]
#[case::empty(0, "")]
#[case::single(1, "a")]
#[case::pair(2, "ab")]
#[case::three(3, "abc")]
#[case::four(4, "abcd")]
fn builder_runs_hooks_in_append_order(
    scripts: Py<PyDict>,
    #[case] count: usize,
    #[case] expected: &str,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = ["first", "second", "third", "fourth"]
            .into_iter()
            .take(count)
            .fold(HookChain::new(), |chain, name| {
                chain.with(hook(py, &scripts, name))
            });
        let original = PyDict::new(py).unbind();
        let result = finish(
            py,
            &task,
            hooks.prepare_arguments(&task, original.clone_ref(py), 0.0),
        )
        .unwrap();
        if count == 0 {
            assert!(result.bind(py).is(original.bind(py)));
        } else {
            assert_eq!(
                result
                    .bind(py)
                    .get_item("order")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                expected,
            );
        }
        assert!(original.bind(py).is_empty());
        let locals = scripts.bind(py);
        locals.set_item("expected", expected).unwrap();
        py.run(
            c"assert ''.join(name for name, kind, value in log) == expected",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

/// The last hook appended is the innermost layer: it sees the response first, so a body
/// hook composed after the wrapper transforms before the wrapper's own success hook.
#[rstest]
#[case::single(1, "a")]
#[case::pair(2, "ba")]
#[case::three(3, "cba")]
fn response_transformations_run_from_the_innermost_hook_outward(
    scripts: Py<PyDict>,
    #[case] count: usize,
    #[case] expected: &str,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let task = PythonTask::new(asynchronous);
        let mut hooks = ["first", "second", "third"]
            .into_iter()
            .take(count)
            .fold(HookChain::new(), |chain, name| {
                chain.with(hook(py, &scripts, name))
            });
        let response = py.None();
        let transformed = finish(
            py,
            &task,
            hooks.transform_response(&task, response.clone_ref(py), TIMING),
        )
        .unwrap();
        let locals = scripts.bind(py);
        locals.set_item("transformed", transformed).unwrap();
        locals.set_item("expected", expected).unwrap();
        py.run(
            c"
applied = []
value = transformed
while isinstance(value, tuple):
    value, name = value
    applied.insert(0, name)
assert value is None
assert ''.join(applied) == expected, applied
",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}
