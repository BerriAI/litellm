use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple};

use crate::constants::{
    AWAIT_ADAPTER_FILENAME, AWAIT_ADAPTER_FUNCTION, AWAIT_ADAPTER_MODULE, AWAIT_ADAPTER_SOURCE,
};

static AWAIT_ADAPTER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// How a retained callback is bound to its caller.
///
/// `Direct` mirrors `callable(*args, **kwargs)`: a coroutine returned by the
/// callable is handed back untouched and never awaited. `Await` mirrors
/// `await callable(*args, **kwargs)` inline in the caller's task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvocationMode {
    Direct,
    Await,
}

/// Result of [`PreparedCall::invoke`].
///
/// `Awaitable` carries an adapter coroutine that has not yet called the
/// callback. The callback runs, and any exception it raises surfaces, only
/// when Python drives that coroutine.
#[derive(Debug)]
pub enum InvocationOutcome {
    Returned(Py<PyAny>),
    Awaitable(Py<PyAny>),
}

/// A callback plus its arguments, retained as owning Python references.
///
/// Arguments are passed to the callback by identity, never copied, so the
/// callback observes and may mutate the caller's objects. Dropping the value
/// releases the references; a Python-visible owner must also expose them to
/// the cycle collector via [`PreparedCall::traverse`].
pub struct PreparedCall {
    mode: InvocationMode,
    callable: Py<PyAny>,
    positional: Py<PyTuple>,
    keywords: Option<Py<PyDict>>,
}

impl PreparedCall {
    pub fn new(
        mode: InvocationMode,
        callable: Py<PyAny>,
        positional: Py<PyTuple>,
        keywords: Option<Py<PyDict>>,
    ) -> Self {
        Self {
            mode,
            callable,
            positional,
            keywords,
        }
    }

    pub fn invoke(&self, py: Python<'_>) -> PyResult<InvocationOutcome> {
        match self.mode {
            InvocationMode::Direct => self
                .callable
                .call(
                    py,
                    self.positional.bind(py),
                    self.keywords.as_ref().map(|kwargs| kwargs.bind(py)),
                )
                .map(InvocationOutcome::Returned),
            InvocationMode::Await => await_adapter(py)?
                .call1(py, (&self.callable, &self.positional, &self.keywords))
                .map(InvocationOutcome::Awaitable),
        }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            mode: self.mode,
            callable: self.callable.clone_ref(py),
            positional: self.positional.clone_ref(py),
            keywords: self.keywords.as_ref().map(|value| value.clone_ref(py)),
        }
    }

    pub fn traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.callable)?;
        visit.call(&self.positional)?;
        if let Some(keywords) = &self.keywords {
            visit.call(keywords)?;
        }
        Ok(())
    }
}

/// Compiling the adapter runs Python, which may re-enter this function through
/// audit hooks. `PyOnceLock` forbids re-entrant initialization, so compile
/// first and only publish a finished adapter into the cell.
fn await_adapter(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    if let Some(adapter) = AWAIT_ADAPTER.get(py) {
        return Ok(adapter);
    }
    let compiled = PyModule::from_code(
        py,
        AWAIT_ADAPTER_SOURCE,
        AWAIT_ADAPTER_FILENAME,
        AWAIT_ADAPTER_MODULE,
    )?
    .getattr(AWAIT_ADAPTER_FUNCTION)?
    .unbind();
    Ok(AWAIT_ADAPTER.get_or_init(py, || compiled))
}
