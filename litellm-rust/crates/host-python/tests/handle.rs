mod support;

use litellm_host::{hooks::CallHooks, lifecycle::Timing};
use litellm_host_python::{
    Awaiter, Execution, ExecutionBody, ExecutionStep, PythonCallEvent, PythonOwned, PythonRuntime,
    Suspension, run_call,
};
use parking_lot::Mutex;
use pyo3::{
    exceptions::PyBaseException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyCFunction, PyDict},
};
use rstest::rstest;
use support::{
    Interpreter, Log, OpScript, binding, call_options, install_lifecycle_module, interpreter,
    lifecycle_binding, success_machine,
};

struct RetainingHost {
    retained: Option<Py<PyAny>>,
}

impl ExecutionBody for RetainingHost {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| Ok(ExecutionStep::Return(py.None())))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.retained)
    }
}

#[pyfunction]
fn retaining_coroutine(py: Python<'_>, retained: Py<PyAny>) -> PyResult<Py<Execution>> {
    Py::new(
        py,
        Execution::new(
            RetainingHost {
                retained: Some(retained),
            },
            lifecycle_binding,
        ),
    )
}

struct AwaitBody(Option<Py<PyAny>>);

impl ExecutionBody for AwaitBody {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        match self.0.take() {
            Some(awaitable) => Ok(ExecutionStep::Suspended(Suspension::Await(awaitable))),
            None => result
                .expect("selected await completed")
                .map(ExecutionStep::Return),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

#[pyfunction]
fn await_execution(awaitable: Py<PyAny>) -> Execution {
    Execution::new(AwaitBody(Some(awaitable)), lifecycle_binding)
}

struct CallingBody(Py<PyAny>);

impl ExecutionBody for CallingBody {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| self.0.call0(py).map(ExecutionStep::Return))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

#[pyfunction]
fn calling_execution(callback: Py<PyAny>) -> Execution {
    Execution::new(CallingBody(callback), lifecycle_binding)
}

struct ErrorBody(Option<Py<PyBaseException>>);

impl ExecutionBody for ErrorBody {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| {
            Err(PyErr::from_value(
                self.0.take().unwrap().into_bound(py).into_any(),
            ))
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}

#[pyfunction]
fn error_execution(error: Bound<'_, PyBaseException>) -> Execution {
    Execution::new(ErrorBody(Some(error.unbind())), lifecycle_binding)
}

/// Where a paused call keeps the test's object: in its hooks, or selected as the call's
/// response or exception, with the call paused in the hook that is told about it.
#[derive(Clone, Copy, Debug)]
enum Holder {
    Hooks,
    Response,
    Exception,
}

struct PausingHooks {
    retained: Mutex<Option<Py<PyAny>>>,
    pause: Mutex<Option<Py<PyAny>>>,
    holder: Holder,
}

impl CallHooks<PythonRuntime> for PausingHooks {
    async fn transform_response<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<Py<PyAny>> {
        let retained = || self.retained.lock().take().unwrap();
        match self.holder {
            Holder::Hooks => Ok(response),
            Holder::Response => Ok(retained()),
            Holder::Exception => Err(Python::attach(|py| {
                PyErr::from_value(retained().into_bound(py))
            })),
        }
    }

    async fn on_event<'a>(
        &'a self,
        task: &'a dyn Awaiter,
        event: PythonCallEvent<'_>,
    ) -> PyResult<()> {
        let pauses_here = match self.holder {
            Holder::Hooks => true,
            Holder::Response => matches!(event, PythonCallEvent::Succeeded { .. }),
            Holder::Exception => matches!(event, PythonCallEvent::Failed { .. }),
        };
        let pause = self.pause.lock().take_if(|_| pauses_here);
        if let Some(pause) = pause {
            task.awaited(pause).await?;
        }
        Ok(())
    }
}

impl PythonOwned for PausingHooks {
    fn close(&mut self, _: Python<'_>) {
        *self.retained.get_mut() = None;
        *self.pause.get_mut() = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for slot in [&self.retained, &self.pause] {
            if let Some(value) = slot.try_lock() {
                visit.call(&*value)?;
            }
        }
        Ok(())
    }
}

fn paused_call(
    py: Python<'_>,
    retained: Py<PyAny>,
    pause: Py<PyAny>,
    holder: Holder,
) -> PyResult<Py<PyAny>> {
    let machine = success_machine();
    run_call(
        py,
        move |_, _, request| Ok(machine(request)),
        binding(&Log::default(), OpScript::Answer),
        PausingHooks {
            retained: Mutex::new(Some(retained)),
            pause: Mutex::new(Some(pause)),
            holder,
        },
        PyDict::new(py).unbind(),
        call_options(true),
    )
}

#[rstest]
fn every_suspension_is_awaited_inline_on_the_callers_task(
    #[from(interpreter)] _python: Interpreter,
) {
    Python::attach(|py| {
        let module = install_lifecycle_module(py);
        let locals = PyDict::new(py);
        locals
            .set_item("drive", module.getattr("drive").unwrap())
            .unwrap();
        locals
            .set_item(
                "await_execution",
                wrap_pyfunction!(await_execution, py).unwrap(),
            )
            .unwrap();
        locals
            .set_item(
                "calling_execution",
                wrap_pyfunction!(calling_execution, py).unwrap(),
            )
            .unwrap();
        let probe = std::ffi::CString::new(include_str!("lifecycle.py")).unwrap();
        py.run(&probe, Some(&locals), Some(&locals)).unwrap();
    });
}

#[rstest]
fn retained_exception_frames_are_collectable(#[from(interpreter)] _python: Interpreter) {
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "error_execution",
                wrap_pyfunction!(error_execution, py).unwrap(),
            )
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    try:
        raise ValueError('retained traceback')
    except ValueError as error:
        retained.owner = error_execution(error)
    return weakref.ref(retained)

reference = cycle()
gc.collect()
assert reference() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[rstest]
fn cycles_through_a_retained_execution_are_collectable(#[from(interpreter)] _python: Interpreter) {
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "retaining_coroutine",
                wrap_pyfunction!(retaining_coroutine, py).unwrap(),
            )
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref

class Retained:
    pass

def cycle():
    retained = Retained()
    coroutine = retaining_coroutine(retained)
    retained.coroutine = coroutine
    return weakref.ref(retained)

retained_ref = cycle()
gc.collect()
assert retained_ref() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}

#[rstest]
#[case::held_by_the_hooks(Holder::Hooks)]
#[case::selected_as_the_response(Holder::Response)]
#[case::selected_as_the_exception(Holder::Exception)]
fn cycles_through_a_call_suspended_in_a_hook_are_collectable(
    #[from(interpreter)] _python: Interpreter,
    #[case] holder: Holder,
) {
    Python::attach(|py| {
        let locals = PyDict::new(py);
        let call = PyCFunction::new_closure(py, None, None, move |args, _| {
            let (retained, pause) = args.extract::<(Py<PyAny>, Py<PyAny>)>()?;
            paused_call(args.py(), retained, pause, holder)
        })
        .unwrap();
        locals.set_item("paused_call", call).unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref

class Retained(Exception):
    pass

class Pause:
    def __await__(self):
        yield

def cycle():
    retained = Retained()
    coroutine = paused_call(retained, Pause())
    coroutine.send(None)
    retained.coroutine = coroutine
    return weakref.ref(retained)

reference = cycle()
gc.collect()
assert reference() is None
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    });
}
