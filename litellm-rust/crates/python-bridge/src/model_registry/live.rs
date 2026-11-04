use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple};

const MODULE: &str = "litellm.rust_bridge.host.model_cost";
const TOKEN_PRICES: [&str; 2] = ["input_cost_per_token", "output_cost_per_token"];

/// The bound `litellm.model_cost`
pub(super) fn current(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    static CURRENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    CURRENT
        .import(py, MODULE, "current")?
        .call0()?
        .cast_into::<PyDict>()
        .map_err(PyErr::from)
}

/// Whether some `litellm.model_cost` rows under the `str` keys are dicts and none of them carries a per-token price.
/// `get_model_info` fills a missing price with `0`, so this tells an unpriced row from a declared zero
#[pyfunction]
pub(crate) fn cost_map_omits_token_price(keys: &Bound<'_, PyTuple>) -> PyResult<bool> {
    let model_cost = current(keys.py())?;
    let mut found_row = false;
    for key in keys.iter().filter(|key| key.is_instance_of::<PyString>()) {
        let Some(row) = model_cost.get_item(&key)? else {
            continue;
        };
        let Ok(row) = row.cast_into::<PyDict>() else {
            continue;
        };
        for price in TOKEN_PRICES {
            if row.contains(price)? {
                return Ok(false);
            }
        }
        found_row = true;
    }
    Ok(found_row)
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
