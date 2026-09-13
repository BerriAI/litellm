use std::panic::{AssertUnwindSafe, catch_unwind};

use litellm_python_interop::panic_to_pyerr;
use pyo3::exceptions::{PyBaseException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;

pub(super) enum ExecutionStep {
    Return(Py<PyAny>),
    Await(Py<PyAny>),
}

pub(super) trait ExecutionBody: Send + Sync {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep>;
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}

enum ExecutionState {
    Created(Box<dyn ExecutionBody>),
    Running,
    Suspended(Box<dyn ExecutionBody>),
    Closed,
}

#[pyclass]
pub(super) struct Execution {
    state: ExecutionState,
}

impl Execution {
    pub(super) fn new(body: impl ExecutionBody + 'static) -> Self {
        Self {
            state: ExecutionState::Created(Box::new(body)),
        }
    }

    fn advance(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<Py<PyAny>> {
        let mut body = {
            let mut execution = slf.borrow_mut();
            match (&execution.state, result.is_some()) {
                (ExecutionState::Created(_), false) | (ExecutionState::Suspended(_), true) => {}
                (ExecutionState::Running, _) => {
                    return Err(PyRuntimeError::new_err("execution is already running"));
                }
                (ExecutionState::Closed, _) => {
                    return Err(PyRuntimeError::new_err("execution is closed"));
                }
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "execution requires start before resume and can only start once",
                    ));
                }
            }
            match std::mem::replace(&mut execution.state, ExecutionState::Running) {
                ExecutionState::Created(body) | ExecutionState::Suspended(body) => body,
                _ => unreachable!(),
            }
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let step = body.resume(result)?;
            let (tag, value, suspended) = match step {
                ExecutionStep::Await(value) => ("Await", value, true),
                ExecutionStep::Return(value) => ("Complete", value, false),
            };
            let step = py
                .import("litellm.rust_bridge.lifecycle")?
                .getattr(tag)?
                .call1((value,))?
                .unbind();
            Ok((step, suspended))
        }))
        .map_err(panic_to_pyerr)
        .and_then(|result| result);
        match outcome {
            Ok((step, true)) if matches!(slf.borrow().state, ExecutionState::Running) => {
                slf.borrow_mut().state = ExecutionState::Suspended(body);
                Ok(step)
            }
            outcome => {
                slf.borrow_mut().state = ExecutionState::Closed;
                drop(body);
                outcome.and_then(|(step, suspended)| {
                    if suspended {
                        Err(PyRuntimeError::new_err(
                            "execution was closed while running",
                        ))
                    } else {
                        Ok(step)
                    }
                })
            }
        }
    }
}

#[pymethods]
impl Execution {
    fn start(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, None)
    }

    fn resume_value(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        value: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, Some(Ok(value)))
    }

    fn resume_error(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        error: Bound<'_, PyBaseException>,
    ) -> PyResult<Py<PyAny>> {
        Self::advance(slf, py, Some(Err(PyErr::from_value(error.into_any()))))
    }

    fn close(slf: &Bound<'_, Self>) {
        let state = std::mem::replace(&mut slf.borrow_mut().state, ExecutionState::Closed);
        drop(state);
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.state {
            ExecutionState::Created(body) | ExecutionState::Suspended(body) => {
                body.traverse(&visit)
            }
            _ => Ok(()),
        }
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        Self::close(slf);
    }
}
