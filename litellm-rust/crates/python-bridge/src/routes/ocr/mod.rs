mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod project;
mod request;
mod value;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register(module)?;
    lifecycle::register(module)
}
