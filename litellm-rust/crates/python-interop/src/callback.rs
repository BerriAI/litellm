use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple};

static AWAIT_CALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[derive(Clone, Copy)]
pub enum InvocationMode {
    Direct,
    Await,
}

#[derive(Debug)]
pub enum InvocationOutcome {
    Returned(Py<PyAny>),
    Awaitable(Py<PyAny>),
}

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
            InvocationMode::Await => {
                let adapter = AWAIT_CALL.get_or_try_init(py, || {
                    PyModule::from_code(
                        py,
                        c"async def invoke_awaited(callable, positional, keywords):
    if keywords is None:
        return await callable(*positional)
    return await callable(*positional, **keywords)
",
                        c"retained_callback.py",
                        c"_retained_callback",
                    )?
                    .getattr("invoke_awaited")
                    .map(Bound::unbind)
                })?;
                adapter
                    .call1(py, (&self.callable, &self.positional, &self.keywords))
                    .map(InvocationOutcome::Awaitable)
            }
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
