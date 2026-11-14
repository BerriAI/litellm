use std::sync::Arc;

use litellm_callbacks::{RouterRef, RouterUsage};
use pyo3::{exceptions::PyTypeError, prelude::*};

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
