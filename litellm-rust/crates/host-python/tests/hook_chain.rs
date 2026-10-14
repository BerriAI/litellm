mod support;

use std::future::{Future, ready};

use futures_util::{FutureExt, future::BoxFuture};
use litellm_host::{
    hooks::{ArgumentHooks, Cancelled, OutcomeHooks, ProviderHooks, ResponseHooks},
    interceptors::{RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};
use litellm_host_python::{
    Awaiter, Hooks, PythonGroup, PythonOwned, PythonSurface, report_unless_cancelled, run_call,
};
use pyo3::{
    exceptions::PyBaseException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};
use rstest::{fixture, rstest};
use support::{
    Interpreter, Log, OpScript, binding, call_options, handed, interpreter, success_machine,
};

struct ScriptHooks {
    object: Py<PyAny>,
    asynchronous: bool,
}

impl ScriptHooks {
    async fn invoke(
        &self,
        task: &dyn Awaiter,
        name: &str,
        value: impl FnOnce(Python<'_>) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let result = Python::attach(|py| {
            let value = value(py)?;
            if self.asynchronous {
                self.object.call_method1(py, "invoke", (name, value))
            } else {
                self.object.call_method1(py, name, (value,))
            }
        })?;
        if self.asynchronous {
            task.awaited(result).await
        } else {
            Ok(result)
        }
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

impl ScriptHooks {
    async fn event(
        &self,
        task: &dyn Awaiter,
        name: &str,
        value: impl FnOnce(Python<'_>) -> Py<PyAny>,
    ) -> Result<(), Cancelled<PyErr>> {
        let result = self
            .invoke(task, "event", |py| {
                Ok(
                    PyTuple::new(py, [name.into_pyobject(py)?.into_any().unbind(), value(py)])?
                        .into_any()
                        .unbind(),
                )
            })
            .await;
        report_unless_cancelled(result.map(drop))
    }
}

impl ArgumentHooks<PythonSurface> for ScriptHooks {
    async fn prepare<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        arguments: Py<PyDict>,
        _: f64,
    ) -> PyResult<Py<PyDict>> {
        let value = self
            .invoke(task, "prepare", |_| Ok(arguments.into_any()))
            .await?;
        Python::attach(|py| Ok(value.into_bound(py).cast_into::<PyDict>()?.unbind()))
    }

    fn prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "adopt", (arguments,)))?;
        Ok(())
    }
}

impl ProviderHooks<PythonSurface> for ScriptHooks {
    async fn before_provider_request<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        Python::attach(|py| self.object.bind(py).setattr("model", &context.model))?;
        let url = self
            .invoke(task, "before", |py| {
                Ok(wire.url.clone().into_pyobject(py)?.into_any().unbind())
            })
            .await?;
        Ok(Box::new(WireRequest {
            url: Python::attach(|py| url.extract(py))?,
            ..*wire
        }))
    }
}

impl ResponseHooks<PythonSurface> for ScriptHooks {
    async fn transform<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<Py<PyAny>> {
        self.invoke(task, "transform", |_| Ok(response)).await
    }

    fn stream_opened(&self, _head: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "stream", (py.None(),)))?;
        Ok(())
    }

    fn stream_chunk(&self, chunk: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| self.object.call_method1(py, "stream", (chunk,)))?;
        Ok(())
    }
}

impl OutcomeHooks<PythonSurface> for ScriptHooks {
    async fn started<'a>(&'a self, task: &'a dyn Awaiter, _: f64) -> Result<(), Cancelled<PyErr>> {
        self.event(task, "started", |py| py.None()).await
    }

    async fn succeeded<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        _: Timing,
        response: &Py<PyAny>,
    ) -> Result<(), Cancelled<PyErr>> {
        self.event(task, "success", |py| response.clone_ref(py))
            .await
    }

    async fn failed<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        _: Timing,
        _: FailureOrigin,
        error: &Py<PyBaseException>,
    ) -> Result<(), Cancelled<PyErr>> {
        self.event(task, "failure", |py| error.clone_ref(py).into_any())
            .await
    }
}

/// Runs every awaitable a hook hands it to completion on a fresh event loop.
struct RunToCompletion;

impl Awaiter for RunToCompletion {
    fn awaited(&self, awaitable: Py<PyAny>) -> BoxFuture<'_, PyResult<Py<PyAny>>> {
        let result = Python::attach(|py| {
            py.import("asyncio")?
                .call_method1("run", (awaitable,))
                .map(Bound::unbind)
        });
        Box::pin(ready(result))
    }
}

fn finish<T>(hook: impl Future<Output = T>) -> T {
    hook.now_or_never()
        .expect("every awaitable runs to completion inline")
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

fn script_hooks(
    py: Python<'_>,
    scripts: &Py<PyDict>,
    name: &str,
    asynchronous: bool,
) -> ScriptHooks {
    ScriptHooks {
        object: scripts.bind(py).get_item(name).unwrap().unwrap().unbind(),
        asynchronous,
    }
}

fn chain(py: Python<'_>, scripts: &Py<PyDict>, asynchronous: bool) -> Hooks {
    ["first", "second"]
        .into_iter()
        .fold(Hooks::new(), |hooks, name| {
            hooks.with(PythonSurface::group(script_hooks(
                py,
                scripts,
                name,
                asynchronous,
            )))
        })
}

fn mounted(hooks: Hooks, group: impl PythonGroup + 'static) -> Hooks {
    hooks.with(PythonSurface::group(group))
}

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

struct PreparedPolicy;

impl ArgumentHooks<PythonSurface> for PreparedPolicy {
    fn prepared(&self, arguments: &Py<PyDict>) -> PyResult<()> {
        Python::attach(|py| arguments.bind(py).set_item("policy", "configured"))
    }
}

impl ProviderHooks<PythonSurface> for PreparedPolicy {}
impl ResponseHooks<PythonSurface> for PreparedPolicy {}
impl OutcomeHooks<PythonSurface> for PreparedPolicy {}

impl PythonOwned for PreparedPolicy {
    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

#[rstest]
#[case::sync(false)]
#[case::suspended(true)]
fn request_stages_feed_forward_and_response_stages_unwind_over_the_final_values(
    scripts: Py<PyDict>,
    #[case] asynchronous: bool,
) {
    Python::attach(|py| {
        let hooks = mounted(chain(py, &scripts, asynchronous), PreparedPolicy);
        let original = PyDict::new(py).unbind();
        let arguments =
            finish(hooks.prepare(&RunToCompletion, original.clone_ref(py), 0.0)).unwrap();
        hooks.prepared(&arguments).unwrap();
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
        let wire = finish(hooks.before_provider_request(&RunToCompletion, wire, &context)).unwrap();
        assert_eq!(wire.url, "urlab");
        let response = PyDict::new(py).unbind().into_any();
        let final_response =
            finish(hooks.transform(&RunToCompletion, response.clone_ref(py), TIMING)).unwrap();
        finish(hooks.succeeded(&RunToCompletion, TIMING, &final_response)).unwrap();
        hooks.stream_opened(&py.None()).unwrap();
        hooks.stream_chunk(&response).unwrap();
        let locals = scripts.bind(py);
        locals.set_item("arguments", arguments).unwrap();
        locals.set_item("original", original).unwrap();
        locals.set_item("response", response).unwrap();
        locals.set_item("final_response", final_response).unwrap();
        py.run(
            c"
assert arguments['order'] == 'ab'
assert original == {}
assert first.adopted is arguments and second.adopted is arguments
assert first.adopted['policy'] == second.adopted['policy'] == 'configured'
assert first.model == second.model == 'model'
assert final_response == ((response, 'b'), 'a')
assert [(name, kind) for name, kind, value in log] == [
    ('a', 'prepare'), ('b', 'prepare'), ('b', 'success'), ('a', 'success'),
    ('b', 'stream'), ('a', 'stream'), ('b', 'stream'), ('a', 'stream'),
]
assert log[2][2] is final_response and log[3][2] is final_response
assert log[6][2] is response and log[7][2] is response
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
        let hooks = mounted(
            chain(py, &scripts, asynchronous),
            script_hooks(py, &scripts, "third", asynchronous),
        );
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
        let error = pyo3::exceptions::PyValueError::new_err("provider failed").into_value(py);
        let response = py.None();
        if failed {
            finish(hooks.failed(&RunToCompletion, TIMING, FailureOrigin::Call, &error)).unwrap();
        } else {
            finish(hooks.succeeded(&RunToCompletion, TIMING, &response)).unwrap();
        }
        locals
            .set_item("selected", if failed { error.into_any() } else { response })
            .unwrap();
        py.run(
            c"
assert [name for name, kind, value in log] == ['c', 'b', 'a']
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
        let hooks = chain(py, &scripts, asynchronous);
        let locals = scripts.bind(py);
        py.run(
            c"first.error = ValueError('prepare failed')",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let result = finish(hooks.prepare(&RunToCompletion, PyDict::new(py).unbind(), 0.0));
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
        let hooks = chain(py, &scripts, asynchronous);
        let locals = scripts.bind(py);
        py.run(
            c"second.error = asyncio.CancelledError()",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let response = py.None();
        let Cancelled(error) =
            finish(hooks.succeeded(&RunToCompletion, TIMING, &response)).unwrap_err();
        assert!(error.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py));
        py.run(
            c"assert len(log) == 1 and log[0][0] == 'b'",
            Some(locals),
            Some(locals),
        )
        .unwrap();
    });
}

#[rstest]
fn suspended_hooks_keep_the_callers_task_and_context(
    #[from(interpreter)] _python: Interpreter,
    scripts: Py<PyDict>,
) {
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
    if name != 'prepare':
        return value
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
        let machine = success_machine();
        let call = handed(
            py,
            run_call(
                py,
                move |_, _, request| Ok(machine(request)),
                binding(&Log::default(), OpScript::Answer),
                chain(py, &scripts, true),
                PyDict::new(py).unbind(),
                call_options(true),
            ),
            true,
        )
        .unwrap();
        locals.set_item("call", call).unwrap();
        py.run(
            c"
async def exercise():
    global caller, thread
    caller = asyncio.current_task()
    thread = threading.get_ident()
    state.set('caller')
    await call
    assert first.adopted['context'] == 'callerxx'
    assert state.get() == 'callerxx'
    assert second.adopted is first.adopted
asyncio.run(exercise())
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
fn argument_groups_run_in_mount_order(
    scripts: Py<PyDict>,
    #[case] count: usize,
    #[case] expected: &str,
    #[values(false, true)] asynchronous: bool,
) {
    Python::attach(|py| {
        let hooks = ["first", "second", "third", "fourth"]
            .into_iter()
            .take(count)
            .fold(Hooks::new(), |chain, name| {
                mounted(chain, script_hooks(py, &scripts, name, asynchronous))
            });
        let original = PyDict::new(py).unbind();
        let result = finish(hooks.prepare(&RunToCompletion, original.clone_ref(py), 0.0)).unwrap();
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
