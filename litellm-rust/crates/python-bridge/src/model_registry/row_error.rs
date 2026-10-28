use litellm_model_catalog::{Located, Rejects};
use pyo3::prelude::*;

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone, Debug)]
pub(crate) struct RowError {
    #[pyo3(get)]
    pub(crate) key: String,
    #[pyo3(get)]
    pub(crate) field: Option<String>,
    #[pyo3(get)]
    pub(crate) kind: String,
    #[pyo3(get)]
    pub(crate) rejects: bool,
    #[pyo3(get)]
    pub(crate) message: String,
}

pub(crate) fn row_error(error: &Located) -> RowError {
    RowError {
        key: error.key().unwrap_or_default(),
        field: error.field(),
        kind: <&'static str>::from(error.error.kind()).to_owned(),
        rejects: error.error.rejects() == Rejects::Row,
        message: litellm_error::Report::new(&error.error).to_string(),
    }
}
