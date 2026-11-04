use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple};

const MODULE: &str = "litellm.rust_bridge.host.model_cost";

/// The bound `litellm.model_cost`
pub(super) fn current(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    static CURRENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    CURRENT
        .import(py, MODULE, "current")?
        .call0()?
        .cast_into::<PyDict>()
        .map_err(PyErr::from)
}

/// `omits_token_price` over the dict rows of `litellm.model_cost` under the `str` keys
#[pyfunction]
pub(crate) fn cost_map_omits_token_price(keys: &Bound<'_, PyTuple>) -> PyResult<bool> {
    let model_cost = current(keys.py())?;
    let rows = keys
        .iter()
        .filter(|key| key.is_instance_of::<PyString>())
        .map(|key| model_cost.get_item(&key))
        .collect::<PyResult<Vec<_>>>()?;
    let rows = rows
        .into_iter()
        .flatten()
        .filter_map(|row| row.cast_into::<PyDict>().ok());
    Ok(litellm_model_catalog::omits_token_price(rows.map(|row| {
        move |name: &str| row.contains(name).unwrap_or(false)
    })))
}

/// The `litellm.model_cost` row under `key` when it is truthy, else `fallback`, the row a registration of `key`
/// merges onto
#[pyfunction]
pub(crate) fn live_row_or<'py>(
    key: &Bound<'py, PyAny>,
    fallback: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = key.py();
    let row = current(py)?.call_method1(intern!(py, "get"), (key,))?;
    Ok(if row.is_truthy()? { row } else { fallback })
}
