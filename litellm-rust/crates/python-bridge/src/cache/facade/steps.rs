//! Async facade operations driven through `litellm.rust_bridge.lifecycle`.

use litellm_host_python::{Execution, ExecutionBody, ExecutionStep};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
    types::PyDict,
};

use super::{Cache, entries};

pub(super) enum Continuation {
    Lookup {
        facade: Py<PyAny>,
        max_age: Py<PyAny>,
    },
    NativeLookup,
    SemanticLookup {
        kwargs: Py<PyDict>,
    },
    Store {
        facade: Py<PyAny>,
    },
}

pub(super) struct AwaitThen {
    awaitable: Option<Py<PyAny>>,
    continuation: Continuation,
}

impl AwaitThen {
    fn finish(&self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match &self.continuation {
            Continuation::Lookup { facade, max_age } => {
                let outcome = result.and_then(|value| {
                    entries::cache_logic(
                        facade.bind(py).cast::<Cache>()?,
                        value.bind(py),
                        max_age.bind(py),
                    )
                    .map(Bound::unbind)
                });
                swallow_lookup_failure(py, outcome)
            }
            Continuation::NativeLookup => swallow_lookup_failure(py, result),
            Continuation::SemanticLookup { kwargs } => {
                let outcome = result.and_then(|value| {
                    let (response, similarity) = value.extract::<(Py<PyAny>, Option<f64>)>(py)?;
                    entries::stamp_semantic_similarity(kwargs.bind(py).as_any(), similarity)?;
                    Ok(response)
                });
                swallow_lookup_failure(py, outcome)
            }
            Continuation::Store { facade } => match result {
                Ok(_) => Ok(py.None()),
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    entries::log_add_cache_failure(facade.bind(py).cast::<Cache>()?, &error)?;
                    Ok(py.None())
                }
                Err(error) => Err(error),
            },
        }
    }
}

fn swallow_lookup_failure(py: Python<'_>, outcome: PyResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    match outcome {
        Ok(value) => Ok(value),
        Err(error) if error.is_instance_of::<PyException>(py) => {
            entries::log_lookup_failure(py, &error)?;
            Ok(py.None())
        }
        Err(error) => Err(error),
    }
}

impl ExecutionBody for AwaitThen {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        Python::attach(|py| match result {
            None => self
                .awaitable
                .take()
                .map(ExecutionStep::Await)
                .ok_or_else(|| PyRuntimeError::new_err("cache execution already started")),
            Some(result) => self.finish(py, result).map(ExecutionStep::Return),
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(awaitable) = &self.awaitable {
            visit.call(awaitable)?;
        }
        match &self.continuation {
            Continuation::Lookup { facade, max_age } => {
                visit.call(facade)?;
                visit.call(max_age)
            }
            Continuation::NativeLookup => Ok(()),
            Continuation::SemanticLookup { kwargs } => visit.call(kwargs),
            Continuation::Store { facade } => visit.call(facade),
        }
    }
}

struct Immediate(Option<Py<PyAny>>);

impl ExecutionBody for Immediate {
    fn resume(&mut self, _result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        self.0
            .take()
            .map(ExecutionStep::Return)
            .ok_or_else(|| PyRuntimeError::new_err("cache execution already finished"))
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.0 {
            Some(value) => visit.call(value),
            None => Ok(()),
        }
    }
}

fn drive<'py>(py: Python<'py>, body: impl ExecutionBody + 'static) -> PyResult<Bound<'py, PyAny>> {
    let execution = Py::new(py, Execution::new(body))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
}

pub(super) fn after<'py>(
    py: Python<'py>,
    awaitable: Bound<'py, PyAny>,
    continuation: Continuation,
) -> PyResult<Bound<'py, PyAny>> {
    drive(
        py,
        AwaitThen {
            awaitable: Some(awaitable.unbind()),
            continuation,
        },
    )
}

pub(super) fn ready_none(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    drive(py, Immediate(Some(py.None())))
}
