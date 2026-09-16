use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(FromPyObject)]
pub(crate) struct PythonLogger(Py<PyAny>);

impl PythonLogger {
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.0.bind(py)
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self(self.0.clone_ref(py))
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }

    pub(super) fn defers_async_logging(&self, py: Python<'_>) -> bool {
        self.object(py)
            .getattr("_defer_async_logging")
            .is_ok_and(|value| value.is_truthy().unwrap_or(false))
    }

    pub(super) fn defer_success(
        &self,
        py: Python<'_>,
        pending: Py<super::DeferredSuccess>,
    ) -> PyResult<()> {
        self.object(py).setattr("_native_pending_logging", pending)
    }
}

pub(super) fn finalize(
    py: Python<'_>,
    response: &Option<Py<PyAny>>,
    logger: &PythonLogger,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    end: &Option<Py<PyAny>>,
) -> PyResult<()> {
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("finalize")?
        .call1((response, logger.object(py), kwargs, start, end))?;
    Ok(())
}

pub(super) fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}

pub(super) struct DeploymentHooks;

impl DeploymentHooks {
    pub(super) fn before_call(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_pre_call_deployment_hook")?
            .call1((kwargs, call_type))
            .map(Bound::unbind)
    }

    pub(super) fn after_success(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        response: &Option<Py<PyAny>>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_success_deployment_hook")?
            .call1((kwargs, response, call_type))
            .map(Bound::unbind)
    }

    pub(super) fn after_failure(
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        error: &Py<PyBaseException>,
        call_type: &str,
    ) -> PyResult<Py<PyAny>> {
        py.import("litellm.utils")?
            .getattr("async_post_call_failure_deployment_hook")?
            .call1((kwargs, error, call_type))
            .map(Bound::unbind)
    }
}
