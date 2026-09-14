mod lifecycle;
mod websocket;

pub(super) fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    websocket::register(module)?;
    lifecycle::register(module)
}
