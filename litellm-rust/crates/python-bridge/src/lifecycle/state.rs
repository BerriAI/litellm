use litellm_core::call_lifecycle::host::{HostPhase, HostStep};
use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::bindings::{self, DeploymentHooks, PythonLogger};
use super::preparation;

pub(crate) fn missing_state() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("missing native call state")
}

pub(crate) struct PythonCallState {
    pub args: Py<PyTuple>,
    pub kwargs: Py<PyDict>,
    pub logger: Option<PythonLogger>,
    pub start: Py<PyAny>,
    pub end: Option<Py<PyAny>>,
    pub response: Option<Py<PyAny>>,
    pub error: Option<Py<PyBaseException>>,
    pub asynchronous: bool,
    pub internal: bool,
    pub call_type: &'static str,
}

pub(crate) fn now(py: Python<'_>) -> PyResult<Py<PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method0("now")
        .map(Bound::unbind)
}

impl PythonCallState {
    pub(super) fn invoke(
        &mut self,
        py: Python<'_>,
        phase: HostPhase,
    ) -> PyResult<HostStep<Py<PyAny>, Py<PyAny>>> {
        match phase {
            HostPhase::Setup => self.setup(py)?,
            HostPhase::DeploymentPreCall => {
                if !DeploymentHooks::needed(py)? {
                    return Ok(HostStep::Ready(self.kwargs.clone_ref(py).into_any()));
                }
                return Ok(HostStep::Suspend(DeploymentHooks::before_call(
                    py,
                    &self.kwargs,
                    self.call_type,
                )?));
            }
            HostPhase::Prepare => self.prepare(py)?,
            HostPhase::DeploymentPostCall => {
                if !DeploymentHooks::needed(py)? {
                    return self
                        .response
                        .as_ref()
                        .map(|value| HostStep::Ready(value.clone_ref(py)))
                        .ok_or_else(missing_state);
                }
                return Ok(HostStep::Suspend(DeploymentHooks::after_success(
                    py,
                    &self.kwargs,
                    &self.response,
                    self.call_type,
                )?));
            }
            HostPhase::Finalize => self.finalize(py)?,
            HostPhase::Success => self.dispatch_success(py)?,
            HostPhase::DeploymentFailure => {
                if let Some(error) = &self.error
                    && DeploymentHooks::needed(py)?
                {
                    return Ok(HostStep::Suspend(DeploymentHooks::after_failure(
                        py,
                        &self.kwargs,
                        error,
                        self.call_type,
                    )?));
                }
            }
            HostPhase::Failure | HostPhase::AsyncFailure => {
                if let Some(awaitable) =
                    self.dispatch_failure(py, phase == HostPhase::AsyncFailure)?
                {
                    return Ok(HostStep::Suspend(awaitable));
                }
            }
            HostPhase::Execute
            | HostPhase::ConstructResponse
            | HostPhase::MapFailure
            | HostPhase::Complete => return Err(missing_state()),
        }
        Ok(HostStep::Ready(py.None()))
    }

    pub(super) fn accept(
        &mut self,
        py: Python<'_>,
        phase: HostPhase,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        match phase {
            HostPhase::DeploymentPreCall => {
                self.kwargs = value.into_bound(py).cast_into::<PyDict>()?.unbind()
            }
            HostPhase::DeploymentPostCall => self.response = Some(value),
            _ => {}
        }
        Ok(())
    }

    pub fn new(
        py: Python<'_>,
        args: Py<PyTuple>,
        kwargs: Py<PyDict>,
        asynchronous: bool,
        call_type: &'static str,
    ) -> PyResult<Self> {
        Ok(Self {
            args,
            kwargs,
            logger: None,
            start: py.None(),
            end: None,
            response: None,
            error: None,
            asynchronous,
            internal: false,
            call_type,
        })
    }

    pub fn logger(&self) -> PyResult<&PythonLogger> {
        self.logger.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("call logging is not initialized")
        })
    }

    pub fn setup(&mut self, py: Python<'_>) -> PyResult<()> {
        self.start = now(py)?;
        self.internal = bindings::is_internal_call(py)?;
        let result = bindings::setup(
            py,
            self.call_type,
            &self.args,
            &self.kwargs,
            &self.start,
            self.asynchronous,
        )?;
        self.logger = Some(result.logger()?);
        self.kwargs = result.kwargs()?;
        Ok(())
    }

    pub fn prepare(&mut self, py: Python<'_>) -> PyResult<()> {
        self.kwargs = preparation::prepare(py, self.kwargs.bind(py), self.logger()?)?.unbind();
        Ok(())
    }

    pub fn finalize(&self, py: Python<'_>) -> PyResult<()> {
        bindings::finalize(
            py,
            &self.response,
            self.logger()?,
            &self.kwargs,
            &self.start,
            &self.end,
        )
    }

    pub fn cleanup(&mut self, py: Python<'_>) {
        if let Some(logger) = self.logger.take()
            && let Err(error) = logger.restore_context(py)
        {
            error.write_unraisable(py, None);
        }
    }

    pub fn retain_error(&mut self, py: Python<'_>, error: PyErr) {
        self.error = Some(error.into_value(py));
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)?;
        if let Some(logger) = &self.logger {
            logger.traverse(visit)?;
        }
        visit.call(&self.start)?;
        visit.call(&self.end)?;
        visit.call(&self.response)?;
        visit.call(&self.error)
    }
}
