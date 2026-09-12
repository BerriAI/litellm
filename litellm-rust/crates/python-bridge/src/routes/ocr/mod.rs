mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod value;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register(module)?;
    document::register(module)?;
    lifecycle::register(module)
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register_trace(module)
}
