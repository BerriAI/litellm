use pyo3::prelude::*;

use crate::errors::RustBridgeDeclined;

#[pyfunction]
fn responses_websocket() -> PyResult<()> {
    Err(RustBridgeDeclined::new_err(
        "Responses WebSocket requires host per-frame guardrails and logging",
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(responses_websocket, module)?)
}
