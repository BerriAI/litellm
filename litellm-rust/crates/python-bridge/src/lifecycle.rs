use pyo3::prelude::*;

/// The product stream module every route binds unless it supplies its own.
pub(crate) fn binding(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("litellm.rust_bridge.streams")
}
