use litellm_host_python::to_py;
use litellm_model_catalog::{ModelRow, Row, RowShape};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};

pub(crate) const MODULE: &str = "litellm.rust_bridge.host.model_cost";

/// Builds a dict holding `rows` in order, aliases sharing their owner's object
pub(crate) fn table<'py>(py: Python<'py>, rows: &[Row<'_>]) -> PyResult<Bound<'py, PyDict>> {
    let table = PyDict::new(py);
    for row in rows {
        let value = match &row.shape {
            RowShape::AliasOf(owner) => table
                .get_item(owner)?
                .expect("an owner is served before its aliases"),
            RowShape::Model(model) => model_row(py, model)?,
            RowShape::Section(value) => to_py(py, value)?.into_bound(py),
        };
        table.set_item(row.key, value)?;
    }
    Ok(table)
}

/// A plain dict, or main's `AliasedCostEntry` for a row with aliases or registered fields
fn model_row<'py>(py: Python<'py>, model: &ModelRow<'_>) -> PyResult<Bound<'py, PyAny>> {
    let fields = to_py(py, model.fields.as_ref())?.into_bound(py);
    if model.names.is_empty() && model.registered_fields.is_empty() {
        return Ok(fields);
    }
    py.import(MODULE)?.getattr("entry")?.call1((
        fields,
        PyTuple::new(py, model.names)?,
        PyFrozenSet::new(py, model.registered_fields.iter().copied())?,
    ))
}

/// Writes one registration into `rows` the way main's `register_model` does and returns the row key it landed on
pub(crate) fn write<'py>(
    rows: &Bound<'py, PyDict>,
    key: &str,
    registration: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&str>,
    warning_display_name: Option<&str>,
) -> PyResult<String> {
    rows.py()
        .import(MODULE)?
        .getattr("write")?
        .call1((
            rows,
            key,
            registration,
            custom_llm_provider,
            warning_display_name,
        ))?
        .extract()
}
