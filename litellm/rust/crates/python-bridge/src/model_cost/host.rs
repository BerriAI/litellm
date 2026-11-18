//! Calls into `litellm.rust_bridge.host.model_cost`

use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{Generation, Inferred, KnownNames, NameHost};
use pyo3::call::PyCallArgs;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyTuple};

use super::live::row_fields;

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

/// The published table, the generation a pricing snapshot of it is keyed on, and the Python-held state it reads
pub(crate) fn generation(py: Python<'_>) -> PyResult<GenerationRead<'_>> {
    let (rows, version, configs): (
        Bound<'_, PyDict>,
        (usize, usize, u64, u64),
        Bound<'_, PyTuple>,
    ) = call(py, "generation", ())?.extract()?;
    let (table, size, writes, rules) = version;
    let (discounts, margins, rule_list) = configs.extract()?;
    Ok(GenerationRead {
        rows,
        generation: Generation {
            table,
            size,
            writes,
            rules,
        },
        discounts,
        margins,
        rules: rule_list,
    })
}

/// The table, its generation, `cost_discount_config`, `cost_margin_config` and the `fallback_generalizations` rules
pub(crate) struct GenerationRead<'py> {
    pub(crate) rows: Bound<'py, PyDict>,
    pub(crate) generation: Generation,
    pub(crate) discounts: Bound<'py, PyAny>,
    pub(crate) margins: Bound<'py, PyAny>,
    pub(crate) rules: Bound<'py, PyAny>,
}

/// The cost map behind `litellm.model_cost`
pub(crate) fn inputs(py: Python<'_>) -> PyResult<Bound<'_, super::NativeModelCostMap>> {
    Ok(call(py, "inputs", ())?.cast_into()?)
}

pub(crate) fn known_names(py: Python<'_>) -> PyResult<KnownNames> {
    let (providers, renamed_models, vertex_renamed, replicate_models) =
        call(py, "known_names", ())?.extract()?;
    Ok(KnownNames {
        providers,
        renamed_models,
        vertex_renamed,
        replicate_models,
    })
}

/// The two answers about a name only Python has: the provider `get_llm_provider` reads off it, and the row
/// `get_model_info` answers for a provider that serves its own. A host call that raises answers as a lookup that
/// misses
pub(crate) struct PythonNames<'py> {
    pub(crate) py: Python<'py>,
}

impl NameHost for PythonNames<'_> {
    fn infer(&self, model: &str) -> Option<Inferred> {
        let found: Option<(String, String)> = call(self.py, "infer_provider", (model,))
            .and_then(|found| found.extract())
            .ok()?;
        found.map(|(model, provider)| Inferred { model, provider })
    }

    fn served_row(&self, model: &str, provider: &str) -> Option<Fields> {
        let served = call(self.py, "served_row", (model, provider)).ok()?;
        row_fields(served.cast::<PyMapping>().ok()?).ok()
    }
}

pub(crate) fn published(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    Ok(call(py, "published", ())?.cast_into::<PyDict>()?)
}

/// Writes `written` into `rows`, the published table, through the host, which tracks the table's version
pub(crate) fn write(rows: &Bound<'_, PyDict>, written: &Bound<'_, PyDict>) -> PyResult<()> {
    call(rows.py(), "write", (rows, written)).map(drop)
}
