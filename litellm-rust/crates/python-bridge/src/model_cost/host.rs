//! Calls into `litellm.rust_bridge.host.model_cost`

use pyo3::call::PyCallArgs;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub(crate) const MODULE: &str = "litellm.rust_bridge.host.model_cost";

/// Where a registration lands in the published table and the row it starts from
#[derive(FromPyObject)]
pub(crate) struct Landing<'py> {
    #[pyo3(attribute)]
    pub(crate) key: String,
    #[pyo3(attribute)]
    pub(crate) base: Bound<'py, PyAny>,
    #[pyo3(attribute)]
    pub(crate) matched_builtin: bool,
}

fn call<'py>(
    py: Python<'py>,
    name: &str,
    args: impl PyCallArgs<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import(MODULE)?.getattr(name)?.call1(args)
}

/// The row object every name in `names` shares, the owner first
pub(crate) fn entry<'py>(
    fields: Bound<'py, PyAny>,
    names: &[String],
) -> PyResult<Bound<'py, PyAny>> {
    let py = fields.py();
    call(py, "entry", (fields, PyTuple::new(py, names)?))
}

/// Where `registration` lands in `rows`, `None` when `rows` is no longer the published table. `rows` of `None` is the
/// published table
pub(crate) fn resolve<'py>(
    rows: Option<&Bound<'py, PyDict>>,
    key: &str,
    registration: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&str>,
) -> PyResult<Option<Landing<'py>>> {
    call(
        registration.py(),
        "resolve",
        (rows, key, registration, custom_llm_provider),
    )?
    .extract()
}

/// The row `registration` writes over `current`
pub(crate) fn merged<'py>(
    current: Option<&Bound<'py, PyAny>>,
    landing: &Landing<'py>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    call(
        registration.py(),
        "merged",
        (current, &landing.base, registration),
    )
}

/// The registration kept for a key: `registration` merged into the one kept before it
pub(crate) fn kept<'py>(
    previous: Option<&Bound<'py, PyDict>>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(call(registration.py(), "kept", (previous, registration))?.cast_into::<PyDict>()?)
}

pub(crate) fn published(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    Ok(call(py, "published", ())?.cast_into::<PyDict>()?)
}

/// Writes `written` into `rows`, the published table, through the host, which tracks the table's version
pub(crate) fn write(rows: &Bound<'_, PyDict>, written: &Bound<'_, PyDict>) -> PyResult<()> {
    call(rows.py(), "write", (rows, written)).map(drop)
}
