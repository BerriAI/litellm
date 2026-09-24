//! Async facade operations driven through `litellm.rust_bridge.lifecycle`.

use litellm_host_python::{Execution, ExecutionBody, ExecutionStep};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyException, PyRuntimeError},
    prelude::*,
    types::PyDict,
};

use super::{Cache, embedding, entries};

#[derive(Clone, Copy)]
pub(super) struct Awaited(());

pub(super) enum Start<'py> {
    Await(Bound<'py, PyAny>, Continuation),
    Done(Bound<'py, PyAny>),
}

impl<'py> Start<'py> {
    pub(super) fn none(py: Python<'py>) -> Self {
        Self::Done(py.None().into_bound(py))
    }
}

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
    Forward,
    Discard,
}

impl Continuation {
    fn finish(&self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Lookup { facade, max_age } => {
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
            Self::NativeLookup => swallow_lookup_failure(py, result),
            Self::SemanticLookup { kwargs } => {
                let outcome = result.and_then(|value| {
                    let (response, similarity) = value.extract::<(Py<PyAny>, Option<f64>)>(py)?;
                    entries::stamp_semantic_similarity(kwargs.bind(py).as_any(), similarity)?;
                    Ok(response)
                });
                swallow_lookup_failure(py, outcome)
            }
            Self::Store { facade } => match result {
                Ok(_) => Ok(py.None()),
                Err(error) => log_store_failure(facade.bind(py).cast::<Cache>()?, error),
            },
            Self::Forward => result,
            Self::Discard => result.map(|_| py.None()),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Lookup { facade, max_age } => {
                visit.call(facade)?;
                visit.call(max_age)
            }
            Self::SemanticLookup { kwargs } => visit.call(kwargs),
            Self::Store { facade } => visit.call(facade),
            Self::NativeLookup | Self::Forward | Self::Discard => Ok(()),
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

fn log_store_failure(facade: &Bound<'_, Cache>, error: PyErr) -> PyResult<Py<PyAny>> {
    let py = facade.py();
    if !error.is_instance_of::<PyException>(py) {
        return Err(error);
    }
    entries::log_add_cache_failure(facade, &error)?;
    Ok(py.None())
}

pub(super) enum Operation {
    GetCache {
        facade: Py<Cache>,
        dynamic: Option<Py<PyAny>>,
        kwargs: Py<PyDict>,
    },
    AddCache {
        facade: Py<Cache>,
        result: Py<PyAny>,
        dynamic: Option<Py<PyAny>>,
        kwargs: Py<PyDict>,
    },
    AddCachePipeline {
        facade: Py<Cache>,
        result: Py<PyAny>,
        dynamic: Option<Py<PyAny>>,
        kwargs: Py<PyDict>,
    },
    BatchCacheWrite {
        facade: Py<Cache>,
        result: Py<PyAny>,
        kwargs: Py<PyDict>,
    },
    Ping {
        facade: Py<Cache>,
    },
    DeleteCacheKeys {
        facade: Py<Cache>,
        keys: Py<PyAny>,
    },
    Disconnect {
        facade: Py<Cache>,
    },
}

impl Operation {
    fn start<'py>(&self, py: Python<'py>, awaited: Awaited) -> PyResult<Start<'py>> {
        match self {
            Self::GetCache {
                facade,
                dynamic,
                kwargs,
            } => entries::async_get_cache(
                awaited,
                facade.bind(py),
                dynamic.as_ref().map(|dynamic| dynamic.bind(py)),
                kwargs.bind(py),
            ),
            Self::AddCache {
                facade,
                result,
                dynamic,
                kwargs,
            } => entries::async_add_cache(
                awaited,
                facade.bind(py),
                result.bind(py),
                dynamic.as_ref().map(|dynamic| dynamic.bind(py)),
                kwargs.bind(py),
            ),
            Self::AddCachePipeline {
                facade,
                result,
                dynamic,
                kwargs,
            } => embedding::async_add_cache_pipeline(
                awaited,
                facade.bind(py),
                result.bind(py),
                dynamic.as_ref().map(|dynamic| dynamic.bind(py)),
                kwargs.bind(py),
            ),
            Self::BatchCacheWrite {
                facade,
                result,
                kwargs,
            } => Ok(Start::Await(
                entries::batch_cache_write(
                    awaited,
                    facade.bind(py),
                    result.bind(py),
                    kwargs.bind(py),
                )?,
                Continuation::Discard,
            )),
            Self::Ping { facade } => entries::ping(awaited, facade.bind(py)),
            Self::DeleteCacheKeys { facade, keys } => {
                entries::delete_cache_keys(awaited, facade.bind(py), keys.bind(py))
            }
            Self::Disconnect { facade } => entries::disconnect(awaited, facade.bind(py)),
        }
    }

    fn recover(&self, py: Python<'_>, error: PyErr) -> PyResult<Py<PyAny>> {
        match self {
            Self::GetCache { .. } => swallow_lookup_failure(py, Err(error)),
            Self::AddCache { facade, .. } | Self::AddCachePipeline { facade, .. } => {
                log_store_failure(facade.bind(py), error)
            }
            Self::BatchCacheWrite { .. }
            | Self::Ping { .. }
            | Self::DeleteCacheKeys { .. }
            | Self::Disconnect { .. } => Err(error),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::GetCache {
                facade,
                dynamic,
                kwargs,
            } => {
                visit.call(facade)?;
                visit.call(dynamic)?;
                visit.call(kwargs)
            }
            Self::AddCache {
                facade,
                result,
                dynamic,
                kwargs,
            }
            | Self::AddCachePipeline {
                facade,
                result,
                dynamic,
                kwargs,
            } => {
                visit.call(facade)?;
                visit.call(result)?;
                visit.call(dynamic)?;
                visit.call(kwargs)
            }
            Self::BatchCacheWrite {
                facade,
                result,
                kwargs,
            } => {
                visit.call(facade)?;
                visit.call(result)?;
                visit.call(kwargs)
            }
            Self::DeleteCacheKeys { facade, keys } => {
                visit.call(facade)?;
                visit.call(keys)
            }
            Self::Ping { facade } | Self::Disconnect { facade } => visit.call(facade),
        }
    }
}

enum Phase {
    Pending(Operation),
    Awaiting(Continuation),
    Finished,
}

struct Deferred(Phase);

impl ExecutionBody for Deferred {
    fn resume(&mut self, result: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        let phase = std::mem::replace(&mut self.0, Phase::Finished);
        Python::attach(|py| match (phase, result) {
            (Phase::Pending(operation), None) => match operation.start(py, Awaited(())) {
                Ok(Start::Await(awaitable, continuation)) => {
                    self.0 = Phase::Awaiting(continuation);
                    Ok(ExecutionStep::Await(awaitable.unbind()))
                }
                Ok(Start::Done(value)) => Ok(ExecutionStep::Return(value.unbind())),
                Err(error) => operation.recover(py, error).map(ExecutionStep::Return),
            },
            (Phase::Awaiting(continuation), Some(result)) => {
                continuation.finish(py, result).map(ExecutionStep::Return)
            }
            _ => Err(PyRuntimeError::new_err(
                "cache execution resumed out of order",
            )),
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.0 {
            Phase::Pending(operation) => operation.traverse(visit),
            Phase::Awaiting(continuation) => continuation.traverse(visit),
            Phase::Finished => Ok(()),
        }
    }
}

pub(super) fn defer(py: Python<'_>, operation: Operation) -> PyResult<Bound<'_, PyAny>> {
    let execution = Py::new(py, Execution::new(Deferred(Phase::Pending(operation))))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
}
