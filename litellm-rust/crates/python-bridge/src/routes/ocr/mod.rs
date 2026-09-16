mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod project;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    document::register(module)?;
    lifecycle::register(module)
}
