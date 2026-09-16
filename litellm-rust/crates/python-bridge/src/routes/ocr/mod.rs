mod callbacks;
mod document;
mod errors;
mod lifecycle;
mod project;
mod request;
#[cfg(feature = "trace-parity")]
mod trace;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    lifecycle::register(module)
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    trace::register(module)
}
