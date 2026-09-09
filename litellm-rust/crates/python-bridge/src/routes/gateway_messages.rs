use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;

#[pyfunction]
fn gateway_messages<'py>(
    py: Python<'py>,
    model_alias: String,
    provider_model: String,
    api_base: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] body: Value,
) -> PyResult<Bound<'py, PyAny>> {
    let future = litellm_ai_gateway::trace_parity::messages_request(
        model_alias,
        provider_model,
        api_base,
        body,
    );
    crate::execution::run_async(
        py,
        crate::function_trace::capture(future),
        core_error_to_pyerr,
    )
}

pub(super) fn register_trace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::definition::add_function(module, wrap_pyfunction!(gateway_messages, module)?)
}
