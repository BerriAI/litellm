use pyo3::prelude::*;

use crate::errors::Error;

#[pyfunction]
fn responses_websocket() -> PyResult<()> {
    Err(
        Error::declined("Responses WebSocket requires host per-frame guardrails and logging")
            .into(),
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(responses_websocket, module)?)
}
