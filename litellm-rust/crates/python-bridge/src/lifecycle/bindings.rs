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
    let model = kwargs.bind(py).get_item("model")?;
    let model = model.filter(|value| value.is_instance_of::<pyo3::types::PyString>());
    py.import("litellm.litellm_core_utils.llm_response_utils.response_metadata")?
        .getattr("update_response_metadata")?
        .call1((response, logger.object(py), model, kwargs, start, end))?;
    Ok(())
}

pub(super) fn is_internal_call(py: Python<'_>) -> PyResult<bool> {
    py.import("litellm._internal_context")?
        .getattr("is_internal_call")?
        .call_method0("get")?
        .extract()
}
