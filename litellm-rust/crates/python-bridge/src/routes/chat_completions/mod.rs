mod lifecycle;
mod value;

use pyo3::prelude::*;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    lifecycle::register(module)?;
    value::register(module)
}

#[cfg(feature = "trace-parity")]
pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    value::register_trace(module)
}
