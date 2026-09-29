use litellm_host::{
    hooks::CallHooks,
    interceptors::{RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};
use litellm_host_python::{HookChain, HookStep, PythonCallEvent, PythonOwned, PythonRuntime};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};
use rstest::{fixture, rstest};

struct ScriptHooks {
    object: Py<PyAny>,
    asynchronous: bool,
    wire: Option<Box<WireRequest>>,
}

impl ScriptHooks {
    fn invoke(&self, py: Python<'_>, name: &str, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if self.asynchronous {
            self.object.call_method1(py, "invoke", (name, value))
        } else {
            self.object.call_method1(py, name, (value,))
        }
    }

    fn arguments(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        Ok(HookStep::Ready(
            result?.into_bound(py).cast_into::<PyDict>()?.unbind(),
        ))
    }

    fn response(
        &mut self,
        _: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        result.map(HookStep::Ready)
    }

    fn notification(
        &mut self,
        _: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, ()>> {
        result.map(|_| HookStep::Ready(()))
    }

    fn request(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        let wire = self.wire.take().unwrap();
        Ok(HookStep::Ready(Box::new(WireRequest {
            url: result?.extract(py)?,
            ..*wire
        })))
    }
}

impl PythonOwned for ScriptHooks {
    fn close(&mut self, py: Python<'_>) {
        self.object = py.None();
        self.wire = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.object)
    }
}

impl CallHooks<PythonRuntime> for ScriptHooks {
    fn prepare_arguments(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        _: f64,
    ) -> PyResult<HookStep<Self, Py<PyDict>>> {
        let value = self.invoke(py, "prepare", arguments.into_any())?;
        if self.asynchronous {
            Ok(HookStep::Await(value, Self::arguments))
        } else {
            self.arguments(py, Ok(value))
        }
    }

    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        self.object.call_method1(py, "adopt", (arguments,))?;
        Ok(())
    }

    fn before_provider_request(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<HookStep<Self, Box<WireRequest>>> {
        self.object.bind(py).setattr("model", &context.model)?;
        let value = self.invoke(
            py,
            "before",
            wire.url.clone().into_pyobject(py)?.into_any().unbind(),
        )?;
        self.wire = Some(wire);
        if self.asynchronous {
            Ok(HookStep::Await(value, Self::request))
        } else {
            self.request(py, Ok(value))
        }
    }

    fn transform_response(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<HookStep<Self, Py<PyAny>>> {
        let value = self.invoke(py, "transform", response)?;
        if self.asynchronous {
            Ok(HookStep::Await(value, Self::response))
        } else {
            self.response(py, Ok(value))
        }
    }

    fn on_event(
        &mut self,
        py: Python<'_>,
        event: PythonCallEvent<'_>,
    ) -> PyResult<HookStep<Self, ()>> {
        let (name, value) = match event {
            PythonCallEvent::Succeeded { response, .. } => ("success", response.clone_ref(py)),
            PythonCallEvent::Failed { error, .. } => {
                ("failure", error.clone_ref(py).into_value(py).into_any())
            }
            PythonCallEvent::Started { .. } => ("started", py.None()),
            PythonCallEvent::Cancelled { .. } => ("cancelled", py.None()),
            PythonCallEvent::Execution(_) => ("provider", py.None()),
        };
        let value = self.invoke(
            py,
            "event",
            PyTuple::new(py, [name.into_pyobject(py)?.into_any().unbind(), value])?
                .into_any()
                .unbind(),
        )?;
        if self.asynchronous {
            Ok(HookStep::Await(value, Self::notification))
        } else {
            self.notification(py, Ok(value))
        }
    }

    fn on_stream_open(&mut self, py: Python<'_>, _head: &Py<PyAny>) -> PyResult<()> {
        self.object.call_method1(py, "stream", (py.None(),))?;
        Ok(())
    }

    fn on_stream_chunk(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        self.object.call_method1(py, "stream", (chunk,))?;
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

fn chain(py: Python<'_>, scripts: &Py<PyDict>, asynchronous: bool) -> HookChain {
    let hook = |name| ScriptHooks {
        object: scripts.bind(py).get_item(name).unwrap().unwrap().unbind(),
        asynchronous,
        wire: None,
    };
    HookChain::new().with(hook("first")).with(hook("second"))
}

fn finish<H, T>(py: Python<'_>, hooks: &mut H, step: HookStep<H, T>) -> PyResult<T> {
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

const TIMING: Timing = Timing {
    start_time: 0.0,
    end_time: 1.0,
};

struct PreparedPolicy;

impl CallHooks<PythonRuntime> for PreparedPolicy {
    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        arguments.bind(py).set_item("policy", "configured")
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
        let mut hooks = chain(py, &scripts, asynchronous).with(PreparedPolicy);
        let original = PyDict::new(py).unbind();
        let step = hooks
            .prepare_arguments(py, original.clone_ref(py), 0.0)
            .unwrap();
        let arguments = finish(py, &mut hooks, step).unwrap();
        hooks.arguments_prepared(py, &arguments).unwrap();
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
        let step = hooks.before_provider_request(py, wire, &context).unwrap();
        assert_eq!(finish(py, &mut hooks, step).unwrap().url, "urlab");
        let response = PyDict::new(py).unbind().into_any();
        let step = hooks
            .transform_response(py, response.clone_ref(py), TIMING)
            .unwrap();
        let final_response = finish(py, &mut hooks, step).unwrap();
        let step = hooks
            .on_event(
                py,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &final_response,
                },
            )
            .unwrap();
        finish(py, &mut hooks, step).unwrap();
        hooks.on_stream_open(py, &py.None()).unwrap();
        hooks.on_stream_chunk(py, &response).unwrap();
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
assert final_response == ((response, 'a'), 'b')
assert [(name, kind) for name, kind, value in log] == [
    ('a', 'prepare'), ('b', 'prepare'), ('a', 'success'), ('b', 'success'),
    ('a', 'stream'), ('b', 'stream'), ('a', 'stream'), ('b', 'stream'),
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
        let mut hooks = chain(py, &scripts, asynchronous).with(ScriptHooks {
            object: scripts
                .bind(py)
                .get_item("third")
                .unwrap()
                .unwrap()
                .unbind(),
            asynchronous,
            wire: None,
        });
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
        let step = hooks.on_event(py, event).unwrap();
        finish(py, &mut hooks, step).unwrap();
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
        let mut hooks = chain(py, &scripts, asynchronous);
        let locals = scripts.bind(py);
        py.run(
            c"first.error = ValueError('prepare failed')",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let result = hooks
            .prepare_arguments(py, PyDict::new(py).unbind(), 0.0)
            .and_then(|step| finish(py, &mut hooks, step));
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
        let mut hooks = chain(py, &scripts, asynchronous);
        let locals = scripts.bind(py);
        py.run(
            c"first.error = asyncio.CancelledError()",
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let response = py.None();
        let result = hooks
            .on_event(
                py,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &response,
                },
            )
            .and_then(|step| finish(py, &mut hooks, step));
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

struct Preparing {
    hooks: HookChain,
    arguments: Option<Py<PyDict>>,
    resume: Option<litellm_host_python::HookResume<HookChain, Py<PyDict>>>,
}

impl litellm_host_python::ExecutionBody for Preparing {
    fn resume(
        &mut self,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<litellm_host_python::ExecutionStep> {
        Python::attach(|py| {
            let step = match self.resume.take() {
                Some(resume) => resume(&mut self.hooks, py, result.unwrap())?,
                None => self
                    .hooks
                    .prepare_arguments(py, self.arguments.take().unwrap(), 0.0)?,
            };
            match step {
                HookStep::Ready(arguments) => {
                    self.hooks.arguments_prepared(py, &arguments)?;
                    Ok(litellm_host_python::ExecutionStep::Return(
                        arguments.into_any(),
                    ))
                }
                HookStep::Await(awaitable, resume) => {
                    self.resume = Some(resume);
                    Ok(litellm_host_python::ExecutionStep::Await(awaitable))
                }
            }
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.arguments)?;
        self.hooks.traverse(visit)
    }
}

fn lifecycle(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let source =
        std::ffi::CString::new(include_str!("../../../../litellm/rust_bridge/lifecycle.py"))
            .unwrap();
    PyModule::from_code(py, &source, c"lifecycle.py", c"hook_chain_test_lifecycle")
}

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
        let hooks = chain(py, &scripts, true);
        let execution = litellm_host_python::Execution::new(
            Preparing {
                hooks,
                arguments: Some(PyDict::new(py).unbind()),
                resume: None,
            },
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

#[rstest]
#[case::first_response(false, false)]
#[case::first_exception(true, false)]
#[case::second_response(false, true)]
#[case::second_exception(true, true)]
fn suspended_notification_cycles_are_collectable(
    scripts: Py<PyDict>,
    #[case] failed: bool,
    #[case] first_ready: bool,
) {
    Python::attach(|py| {
        let hook = |name, asynchronous| ScriptHooks {
            object: scripts.bind(py).get_item(name).unwrap().unwrap().unbind(),
            asynchronous,
            wire: None,
        };
        let owner = Py::new(
            py,
            HookOwner {
                hooks: HookChain::new()
                    .with(hook("first", !first_ready))
                    .with(hook("second", true)),
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
        let step = if failed {
            owner.borrow_mut(py).hooks.on_event(
                py,
                PythonCallEvent::Failed {
                    timing: TIMING,
                    origin: FailureOrigin::Call,
                    error: &PyErr::from_value(payload.bind(py).clone()),
                },
            )
        } else {
            owner.borrow_mut(py).hooks.on_event(
                py,
                PythonCallEvent::Succeeded {
                    timing: TIMING,
                    response: &payload,
                },
            )
        }
        .unwrap();
        let HookStep::Await(awaitable, _) = step else {
            panic!("notification must suspend")
        };
        awaitable.call_method0(py, "close").unwrap();
        drop(awaitable);
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
        let mut hooks = ["first", "second", "third", "fourth"]
            .into_iter()
            .take(count)
            .fold(HookChain::new(), |chain, name| {
                chain.with(ScriptHooks {
                    object: scripts.bind(py).get_item(name).unwrap().unwrap().unbind(),
                    asynchronous,
                    wire: None,
                })
            });
        let original = PyDict::new(py).unbind();
        let step = hooks
            .prepare_arguments(py, original.clone_ref(py), 0.0)
            .unwrap();
        let result = finish(py, &mut hooks, step).unwrap();
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
