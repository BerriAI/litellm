use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn bundled_model_cost_map(py: Python<'_>) -> PyResult<Bound<'_, PyBytes>> {
    static BYTES: PyOnceLock<Py<PyBytes>> = PyOnceLock::new();
    BYTES
        .get_or_try_init(py, || -> PyResult<Py<PyBytes>> {
            let bytes = py
                .import("importlib.resources")?
                .call_method1("files", ("litellm",))?
                .call_method1("joinpath", ("model_prices_and_context_window_backup.json",))?
                .call_method0("read_bytes")?
                .cast_into::<PyBytes>()
                .map_err(PyErr::from)?;
            Ok(bytes.unbind())
        })
        .map(|bytes| bytes.bind(py).clone())
}
