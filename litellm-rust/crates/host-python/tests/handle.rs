mod support;

use litellm_host_python::{Execution, ExecutionBody, ExecutionStep};
use pyo3::{
    exceptions::PyBaseException,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use rstest::rstest;
use support::{Interpreter, install_lifecycle_module, interpreter, lifecycle_binding};

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
            Some(awaitable) => Ok(ExecutionStep::Await(awaitable)),
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
