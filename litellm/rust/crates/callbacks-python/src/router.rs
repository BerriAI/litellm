use std::sync::Arc;

use litellm_callbacks::{RemainingUsage, RouterReadError, RouterRef, RouterUsage, UsageAmount};
use litellm_host_python::run_async_value;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::PyDict,
};

#[pyclass(frozen, name = "RustRouterRef")]
pub struct PyRouterRef(RouterRef);

impl PyRouterRef {
    pub fn new(router: Arc<dyn RouterUsage>) -> Self {
        Self(RouterRef::new(router))
    }

    pub fn router(&self) -> &RouterRef {
        &self.0
    }
}

#[pymethods]
impl PyRouterRef {
    fn remaining<'py>(&self, py: Python<'py>, model_group: &str) -> PyResult<Bound<'py, PyAny>> {
        let read = self.0.remaining(model_group);
        run_async_value(py, async move {
            let remaining = read.await.map_err(raised)?;
            Python::attach(|py| headers(py, &remaining).map(Bound::unbind))
        })
    }
}

pub fn raised(error: RouterReadError) -> PyErr {
    match error.downcast::<PyErr>() {
        Ok(error) => *error,
        Err(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

pub fn headers<'py>(py: Python<'py>, remaining: &RemainingUsage) -> PyResult<Bound<'py, PyDict>> {
    let answer = PyDict::new(py);
    for (name, amount) in &remaining.headers {
        match amount {
            UsageAmount::Int(value) => answer.set_item(name, value)?,
            UsageAmount::Float(value) => answer.set_item(name, value)?,
        }
    }
    Ok(answer)
}

pub fn reference(value: &Bound<'_, PyAny>) -> PyResult<Option<RouterRef>> {
    if value.is_none() {
        return Ok(None);
    }
    value
        .cast::<PyRouterRef>()
        .map(|router| Some(router.get().0.clone()))
        .map_err(|_| {
            let type_name = value
                .get_type()
                .name()
                .map_or_else(|_| "object".to_owned(), |name| name.to_string());
            PyTypeError::new_err(format!(
                "router_ref must be a router reference, got {type_name}"
            ))
        })
}
