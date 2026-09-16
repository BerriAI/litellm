mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod project;
mod value;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register(module)?;
    document::register(module)?;
    lifecycle::register(module)
}
