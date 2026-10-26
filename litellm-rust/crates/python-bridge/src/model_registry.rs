mod live;
mod refresh;

pub(crate) use live::*;
pub(crate) use refresh::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use litellm_host_python::{from_py, release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    ApplicationWrite, BUNDLED_REGISTRY, Catalog, Error, IntegrityLimits, Layers, Located,
    SparseRow, diff_row,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyBytes, PyDict, PyFloat, PyFrozenSet, PyList, PyTuple};
use serde::Serialize;
use strum::{IntoStaticStr, VariantArray};

static BUNDLED: LazyLock<ParsedRegistry> = LazyLock::new(|| {
    let loaded = Catalog::parse(BUNDLED_REGISTRY).expect("bundled model registry is valid");
    ParsedRegistry {
        catalog: Arc::new(loaded.value),
        errors: Arc::from(loaded.errors),
    }
});

pub(crate) struct ParsedRegistry {
    pub(crate) catalog: Arc<Catalog>,
    pub(crate) errors: Arc<[Located]>,
}

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum RowErrorKind {
    #[strum(serialize = "json")]
    Json,
    #[strum(serialize = "empty")]
    Empty,
    #[strum(serialize = "below_minimum")]
    BelowMinimum,
    #[strum(serialize = "shrunk")]
    Shrunk,
    #[strum(serialize = "not_an_object")]
    NotAnObject,
    #[strum(serialize = "invalid")]
    Invalid,
    #[strum(serialize = "unknown_block_key")]
    UnknownBlockKey,
    #[strum(serialize = "missing_provider")]
    MissingProvider,
    #[strum(serialize = "unknown_timezone")]
    UnknownTimezone,
    #[strum(serialize = "alias_shadows_model")]
    AliasShadowsModel,
    #[strum(serialize = "alias_taken")]
    AliasTaken,
    #[strum(serialize = "duplicate_price")]
    DuplicatePrice,
    #[strum(serialize = "suffix_inside_block")]
    SuffixInsideBlock,
    #[strum(serialize = "tier_range")]
    TierRange,
    #[strum(serialize = "tier_reversed")]
    TierReversed,
    #[strum(serialize = "tier_gap")]
    TierGap,
    #[strum(serialize = "mixed_tier_basis")]
    MixedTierBasis,
    #[strum(serialize = "not_priced_by_results")]
    NotPricedByResults,
    #[strum(serialize = "bands_from_two_sources")]
    BandsFromTwoSources,
    #[strum(serialize = "not_an_off_peak_rate")]
    NotAnOffPeakRate,
    #[strum(serialize = "no_off_peak_prices")]
    NoOffPeakPrices,
    #[strum(serialize = "no_off_peak_hours")]
    NoOffPeakHours,
    #[strum(serialize = "empty_list")]
    EmptyList,
    #[strum(serialize = "invalid_date")]
    InvalidDate,
}

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone)]
pub(crate) struct RowError {
    #[pyo3(get)]
    pub(crate) key: String,
    #[pyo3(get)]
    pub(crate) field: Option<String>,
    #[pyo3(get)]
    pub(crate) kind: String,
    #[pyo3(get)]
    pub(crate) rejects: bool,
    #[pyo3(get)]
    pub(crate) message: String,
}

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone)]
pub(crate) struct ModelRegistry {
    pub(crate) parsed: Arc<ParsedRegistry>,
    #[pyo3(get)]
    pub(crate) source_revision: Option<String>,
    #[pyo3(get)]
    pub(crate) source_validator: Option<String>,
}

#[pymethods]
impl ModelRegistry {
    #[staticmethod]
    fn parse(py: Python<'_>, body: PyBackedBytes) -> PyResult<Self> {
        let parsed = release_gil(py, move || {
            Catalog::parse(&body).map(|loaded| ParsedRegistry {
                catalog: Arc::new(loaded.value),
                errors: Arc::from(loaded.errors),
            })
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            parsed: Arc::new(parsed),
            source_revision: None,
            source_validator: None,
        })
    }

    #[staticmethod]
    fn bundled(py: Python<'_>) -> Self {
        release_gil(py, || Self {
            parsed: Arc::new(ParsedRegistry {
                catalog: Arc::clone(&BUNDLED.catalog),
                errors: Arc::clone(&BUNDLED.errors),
            }),
            source_revision: None,
            source_validator: None,
        })
    }

    #[staticmethod]
    fn bundled_bytes(py: Python<'_>) -> Bound<'_, PyBytes> {
        PyBytes::new(py, BUNDLED_REGISTRY)
    }

    fn errors(&self) -> Vec<RowError> {
        self.parsed.errors.iter().map(row_error).collect()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &WrittenDocument(self.parsed.catalog.as_ref()))
    }

    fn served(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let catalog = self.parsed.catalog.as_ref();
        let groups = canonical_groups(catalog);
        let rows = PyDict::new(py);
        let mut converted = HashMap::<String, Vec<(Fields, Py<PyDict>)>>::new();
        for (key, fields) in catalog.served() {
            let canonical = groups.get(key).map_or(key, |canonical| canonical.as_str());
            let fields = fields_without_aliases(fields);
            let value = converted.get(canonical).and_then(|projections| {
                projections.iter().find_map(|(canonical_fields, row)| {
                    (canonical_fields == &fields).then(|| row.clone_ref(py))
                })
            });
            let value = match value {
                Some(value) => value,
                None => {
                    let value = fields_to_dict(py, &fields)?;
                    converted
                        .entry(canonical.to_owned())
                        .or_default()
                        .push((fields, value.clone_ref(py)));
                    value
                }
            };
            rows.set_item(key, value)?;
        }
        Ok(rows.unbind().into_any())
    }

    fn fallback_rules(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &self.parsed.catalog.fallback_rules())
    }

    fn check_size(&self, reference: usize) -> Option<RowError> {
        self.parsed
            .catalog
            .check_size(reference, IntegrityLimits::DEFAULT)
            .err()
            .as_ref()
            .map(row_error)
    }

    fn without_rejected(&self) -> (Self, Vec<RowError>) {
        let (catalog, removed) = (*self.parsed.catalog)
            .clone()
            .without_rejected(&self.parsed.errors);
        let removed_errors = removed.into_iter().map(row_error).collect();
        (
            Self {
                parsed: Arc::new(ParsedRegistry {
                    catalog: Arc::new(catalog),
                    errors: Arc::clone(&self.parsed.errors),
                }),
                source_revision: self.source_revision.clone(),
                source_validator: self.source_validator.clone(),
            },
            removed_errors,
        )
    }
}

struct WrittenDocument<'a>(&'a Catalog);

impl Serialize for WrittenDocument<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.written())
    }
}

pub(crate) fn row_error(error: &Located) -> RowError {
    let mut path = error.path.split('/').skip(1);
    let key = unescape_pointer_segment(path.next().unwrap_or_default());
    let field = path.map(unescape_pointer_segment).collect::<Vec<_>>();
    let field = (!field.is_empty()).then(|| field.join("/"));
    let kind = error_kind(&error.error);
    RowError {
        key,
        field,
        kind: <&'static str>::from(kind).to_owned(),
        rejects: litellm_model_catalog::rejects_row(&error.error),
        message: error.error.to_string(),
    }
}

fn unescape_pointer_segment(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn error_kind(error: &Error) -> RowErrorKind {
    match error {
        Error::Json(_) => RowErrorKind::Json,
        Error::Empty => RowErrorKind::Empty,
        Error::BelowMinimum { .. } => RowErrorKind::BelowMinimum,
        Error::Shrunk { .. } => RowErrorKind::Shrunk,
        Error::NotAnObject => RowErrorKind::NotAnObject,
        Error::Invalid(_) => RowErrorKind::Invalid,
        Error::UnknownBlockKey => RowErrorKind::UnknownBlockKey,
        Error::MissingProvider => RowErrorKind::MissingProvider,
        Error::UnknownTimezone { .. } => RowErrorKind::UnknownTimezone,
        Error::AliasShadowsModel { .. } => RowErrorKind::AliasShadowsModel,
        Error::AliasTaken { .. } => RowErrorKind::AliasTaken,
        Error::DuplicatePrice { .. } => RowErrorKind::DuplicatePrice,
        Error::SuffixInsideBlock { .. } => RowErrorKind::SuffixInsideBlock,
        Error::TierRange => RowErrorKind::TierRange,
        Error::TierReversed => RowErrorKind::TierReversed,
        Error::TierGap { .. } => RowErrorKind::TierGap,
        Error::MixedTierBasis => RowErrorKind::MixedTierBasis,
        Error::NotPricedByResults => RowErrorKind::NotPricedByResults,
        Error::BandsFromTwoSources => RowErrorKind::BandsFromTwoSources,
        Error::NotAnOffPeakRate => RowErrorKind::NotAnOffPeakRate,
        Error::NoOffPeakPrices => RowErrorKind::NoOffPeakPrices,
        Error::NoOffPeakHours => RowErrorKind::NoOffPeakHours,
        Error::EmptyList => RowErrorKind::EmptyList,
        Error::InvalidDate => RowErrorKind::InvalidDate,
    }
}

#[pyfunction]
pub(crate) fn model_row_errors(key: &str, row: &Bound<'_, PyAny>) -> PyResult<Vec<RowError>> {
    if !row.is_instance_of::<PyDict>() {
        return Ok(vec![row_error(&Located::new(
            format!("/{}", escape_pointer_segment(key)),
            Error::NotAnObject,
        ))]);
    }
    let mut errors = Vec::new();
    if nonfinite_errors(
        row,
        &format!("/{}", escape_pointer_segment(key)),
        &mut errors,
        0,
    )
    .is_err()
    {
        errors.push(row_error(&Located::new(
            format!("/{}", escape_pointer_segment(key)),
            Error::Invalid(<serde_json::Error as serde::de::Error>::custom(
                "row must contain JSON-compatible values",
            )),
        )));
    }
    if !errors.is_empty() {
        return Ok(errors);
    }
    let fields: Fields = match from_py(row) {
        Ok(fields) => fields,
        Err(_) => {
            let error = <serde_json::Error as serde::de::Error>::custom(
                "row must contain JSON-compatible values",
            );
            return Ok(vec![row_error(&Located::new(
                format!("/{}", escape_pointer_segment(key)),
                Error::Invalid(error),
            ))]);
        }
    };
    Ok(Catalog::row_errors(key, &fields)
        .iter()
        .map(row_error)
        .collect())
}

fn nonfinite_errors(
    value: &Bound<'_, PyAny>,
    path: &str,
    errors: &mut Vec<RowError>,
    depth: usize,
) -> PyResult<()> {
    if depth >= 128 {
        return Err(PyValueError::new_err("model cost row nesting is too deep"));
    }
    if let Ok(number) = value.cast::<PyFloat>() {
        if !number.value().is_finite() {
            errors.push(row_error(&Located::new(
                path.to_owned(),
                Error::Invalid(<serde_json::Error as serde::de::Error>::custom(
                    "JSON numbers must be finite",
                )),
            )));
        }
    } else if let Ok(fields) = value.cast::<PyDict>() {
        for (key, item) in fields.iter() {
            let key: String = key.extract()?;
            nonfinite_errors(
                &item,
                &format!("{path}/{}", escape_pointer_segment(&key)),
                errors,
                depth + 1,
            )?;
        }
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        for (index, item) in value.try_iter()?.enumerate() {
            nonfinite_errors(&item?, &format!("{path}/{index}"), errors, depth + 1)?;
        }
    }
    Ok(())
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct ModelCostLayers {
    layers: Mutex<Layers>,
}

#[pymethods]
impl ModelCostLayers {
    #[new]
    fn new() -> Self {
        Self {
            layers: Mutex::default(),
        }
    }

    fn record_registration(
        &self,
        py: Python<'_>,
        key: String,
        inherits_from: Option<String>,
        fields: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let fields: SparseRow = from_py(fields)?;
        release_gil(py, || {
            self.lock().record(ApplicationWrite::Registration {
                key,
                inherits_from,
                fields,
            })
        });
        Ok(())
    }

    fn diff(&self, py: Python<'_>, changed: &Bound<'_, PyDict>) -> PyResult<()> {
        let changed: BTreeMap<String, (Option<Fields>, Option<Fields>)> = from_py(changed)?;
        release_gil(py, || {
            let mut layers = self.lock();
            for (key, (installed, live)) in changed {
                if let Some(write) = diff_row(&key, installed.as_ref(), live.as_ref()) {
                    layers.record(write);
                }
            }
        });
        Ok(())
    }

    fn forget_writes(&self, py: Python<'_>) {
        release_gil(py, || self.lock().forget_writes());
    }

    fn compose(&self, py: Python<'_>, base: &ModelRegistry) -> PyResult<Py<PyAny>> {
        let rows = release_gil(py, || {
            self.lock()
                .compose(&base.parsed.catalog)
                .map(|row| (row.key.to_owned(), Some((row.fields, row.registered))))
                .collect()
        });
        composed_rows_to_py(py, &base.parsed.catalog, rows)
    }

    fn compose_keys(
        &self,
        py: Python<'_>,
        base: &ModelRegistry,
        keys: BTreeSet<String>,
    ) -> PyResult<Py<PyAny>> {
        let rows = release_gil(py, || {
            self.lock()
                .compose_keys(&base.parsed.catalog, &keys)
                .zip(&keys)
                .map(|(row, key)| (key.clone(), row.map(|row| (row.fields, row.registered))))
                .collect()
        });
        composed_rows_to_py(py, &base.parsed.catalog, rows)
    }
}

impl ModelCostLayers {
    fn lock(&self) -> std::sync::MutexGuard<'_, Layers> {
        self.layers.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn fields_without_aliases(fields: &Fields) -> Fields {
    let mut fields = fields.clone();
    fields.shift_remove("aliases");
    fields
}

fn canonical_groups(catalog: &Catalog) -> HashMap<String, String> {
    catalog
        .served()
        .map(|(key, _)| {
            (
                key.to_owned(),
                catalog
                    .lookup(key)
                    .map_or(key, |matched| matched.canonical_key)
                    .to_owned(),
            )
        })
        .collect()
}

fn fields_to_dict(py: Python<'_>, fields: &Fields) -> PyResult<Py<PyDict>> {
    let row = PyDict::new(py);
    for (key, value) in fields {
        row.set_item(key, to_py(py, value)?)?;
    }
    Ok(row.unbind())
}

type ComposedFields = (String, Option<(Fields, BTreeSet<String>)>);
type ConvertedRows = HashMap<String, Vec<(Fields, BTreeSet<String>, Py<PyDict>)>>;

fn composed_rows_to_py(
    py: Python<'_>,
    catalog: &Catalog,
    rows: Vec<ComposedFields>,
) -> PyResult<Py<PyAny>> {
    let values = PyList::empty(py);
    let mut converted = ConvertedRows::new();
    for (key, row) in rows {
        let group = catalog
            .lookup(&key)
            .filter(|matched| matched.matched_key == key)
            .map_or(key.as_str(), |matched| matched.canonical_key)
            .to_owned();
        let name = key.into_pyobject(py)?.into_any();
        let (fields, registered) = match row {
            Some((fields, registered)) => {
                let fields = fields_without_aliases(&fields);
                let shared = converted.get(&group).and_then(|rows| {
                    rows.iter().find_map(|(previous, owned, value)| {
                        (previous == &fields && owned == &registered).then(|| value.clone_ref(py))
                    })
                });
                let value = match shared {
                    Some(value) => value,
                    None => {
                        let value = fields_to_dict(py, &fields)?;
                        converted.entry(group).or_default().push((
                            fields,
                            registered.clone(),
                            value.clone_ref(py),
                        ));
                        value
                    }
                };
                (
                    value.into_bound(py).into_any(),
                    PyFrozenSet::new(py, registered)?.into_any(),
                )
            }
            None => (py.None().into_bound(py), PyFrozenSet::empty(py)?.into_any()),
        };
        values.append(PyTuple::new(py, [name, fields, registered])?)?;
    }
    Ok(values.unbind().into_any())
}

#[pyfunction]
pub(crate) fn write_rows(
    target: &Bound<'_, PyDict>,
    set: &Bound<'_, PyDict>,
    remove: Vec<String>,
) -> PyResult<()> {
    let updates = set
        .iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value)))
        .collect::<PyResult<Vec<_>>>()?;
    // Keep replaced objects alive until the batch is complete, including Python finalizers.
    let replaced = updates
        .iter()
        .map(|(key, _)| key)
        .chain(&remove)
        .map(|key| target.get_item(key))
        .collect::<PyResult<Vec<_>>>()?;
    for (key, value) in updates {
        target.set_item(key, value)?;
    }
    for key in remove {
        if target.contains(&key)? {
            target.del_item(key)?;
        }
    }
    drop(replaced);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    #[fixture]
    fn scope() -> Py<PyDict> {
        Python::initialize();
        Python::attach(|py| {
            let native = PyModule::new(py, "native").unwrap();
            native.add_class::<ModelRegistry>().unwrap();
            native.add_class::<ModelCostLayers>().unwrap();
            native.add_class::<RowError>().unwrap();
            native
                .add_function(wrap_pyfunction!(model_row_errors, &native).unwrap())
                .unwrap();
            native
                .add_function(wrap_pyfunction!(write_rows, &native).unwrap())
                .unwrap();
            let scope = PyDict::new(py);
            scope.set_item("native", native).unwrap();
            scope.unbind()
        })
    }

    fn json_error() -> serde_json::Error {
        serde_json::from_str::<Value>("{").unwrap_err()
    }

    #[rstest]
    #[case::json(Error::Json(json_error()), "json", false)]
    #[case::empty(Error::Empty, "empty", false)]
    #[case::minimum(Error::BelowMinimum { actual: 1, minimum: 100 }, "below_minimum", false)]
    #[case::shrunk(Error::Shrunk { actual: 100, reference: 300, ratio: 0.5 }, "shrunk", false)]
    #[case::object(Error::NotAnObject, "not_an_object", true)]
    #[case::invalid(Error::Invalid(json_error()), "invalid", true)]
    #[case::block(Error::UnknownBlockKey, "unknown_block_key", true)]
    #[case::provider(Error::MissingProvider, "missing_provider", false)]
    #[case::timezone(Error::UnknownTimezone { zone: "bad".into() }, "unknown_timezone", true)]
    #[case::shadow(Error::AliasShadowsModel { alias: "m".into() }, "alias_shadows_model", true)]
    #[case::taken(Error::AliasTaken { alias: "a".into(), taken_by: "m".into() }, "alias_taken", true)]
    #[case::duplicate(Error::DuplicatePrice { first: "a".into(), second: "b".into() }, "duplicate_price", true)]
    #[case::suffix(Error::SuffixInsideBlock { block: "tiered_pricing" }, "suffix_inside_block", true)]
    #[case::range(Error::TierRange, "tier_range", true)]
    #[case::reversed(Error::TierReversed, "tier_reversed", true)]
    #[case::gap(Error::TierGap { start: 3, expected: 2 }, "tier_gap", true)]
    #[case::basis(Error::MixedTierBasis, "mixed_tier_basis", true)]
    #[case::results(Error::NotPricedByResults, "not_priced_by_results", true)]
    #[case::bands(Error::BandsFromTwoSources, "bands_from_two_sources", true)]
    #[case::rate(Error::NotAnOffPeakRate, "not_an_off_peak_rate", true)]
    #[case::prices(Error::NoOffPeakPrices, "no_off_peak_prices", true)]
    #[case::hours(Error::NoOffPeakHours, "no_off_peak_hours", true)]
    #[case::list(Error::EmptyList, "empty_list", true)]
    #[case::date(Error::InvalidDate, "invalid_date", true)]
    fn row_error_matches_catalog_policy(
        scope: Py<PyDict>,
        #[case] error: Error,
        #[case] kind: &str,
        #[case] rejects: bool,
    ) {
        assert_eq!(litellm_model_catalog::rejects_row(&error), rejects);
        let message = error.to_string();
        Python::attach(|py| {
            let scope = scope.bind(py);
            scope
                .set_item(
                    "error",
                    row_error(&Located::new("/team~1model~0/tiers/0/input", error)),
                )
                .unwrap();
            scope.set_item("kind", kind).unwrap();
            scope.set_item("rejects", rejects).unwrap();
            scope.set_item("message", message).unwrap();
            py.run(
                c_str!(
                    r#"
assert (error.key, error.field, error.kind, error.rejects, error.message) == (
    "team/model~", "tiers/0/input", kind, rejects, message)
try:
    error.kind = "invalid"
except AttributeError:
    pass
else:
    raise AssertionError("row errors must be immutable")
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    fn registry_preserves_written_sections_and_shares_served_aliases(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(
                c_str!(
                    r#"
import json
document = {
    "sample_spec": {"description": "not a model"},
    "model": {"litellm_provider": "openai", "aliases": ["alias"], "input_cost_per_token": 0},
    "bad": {"litellm_provider": "openai", "input_cost_per_token": "invalid"},
    "no-provider": {"input_cost_per_token": 1},
    "non-object": 42,
}
registry = native.ModelRegistry.parse(json.dumps(document).encode())
assert registry.to_dict() == document
assert registry.fallback_rules() is None
served = registry.served()
assert set(served) == {"model", "alias", "bad", "no-provider"}
assert served["model"] is served["alias"]
assert served["model"] == {"litellm_provider": "openai", "input_cost_per_token": 0}
assert {(e.key, e.kind, e.rejects) for e in registry.errors()} == {
    ("bad", "invalid", True), ("no-provider", "missing_provider", False),
    ("non-object", "not_an_object", True),
}
clean, rejected = registry.without_rejected()
assert {e.key for e in rejected} == {"bad", "non-object"}
assert set(clean.served()) == {"model", "alias", "no-provider"}
assert "bad" in registry.to_dict()
assert registry.check_size(0).kind == "below_minimum"
assert registry.check_size(0).key == ""
assert registry.check_size(0).field is None
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    #[case::json(b"{")]
    #[case::root(b"[]")]
    #[case::empty(b"{}")]
    fn fatal_parse_errors_are_value_errors(scope: Py<PyDict>, #[case] body: &[u8]) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            scope.set_item("body", PyBytes::new(py, body)).unwrap();
            py.run(
                c_str!(
                    r#"
try:
    native.ModelRegistry.parse(body)
except ValueError as error:
    assert str(error)
else:
    raise AssertionError("expected fatal catalog parse error")
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    fn bundled_bytes_and_registry_agree(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(
                c_str!(
                    r#"
import json
bundled = native.ModelRegistry.bundled()
body = native.ModelRegistry.bundled_bytes()
assert isinstance(body, bytes)
assert bundled.to_dict() == json.loads(body)
assert bundled.served() == native.ModelRegistry.parse(body).served()
assert bundled.fallback_rules() == bundled.to_dict()["fallback_generalizations"]["rules"]
assert bundled.check_size(0) is None
assert bundled.check_size(len(bundled.served()) * 10).kind == "shrunk"
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::positive_infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    fn nonfinite_values_keep_their_nested_paths(scope: Py<PyDict>, #[case] number: f64) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            scope.set_item("number", number).unwrap();
            py.run(
                c_str!(
                    r#"
errors = native.model_row_errors("team/model~", {
    "litellm_provider": "openai",
    "tiered_pricing": [{"input_cost_per_token": number}],
})
assert [(e.key, e.field, e.kind, e.rejects) for e in errors] == [
    ("team/model~", "tiered_pricing/0/input_cost_per_token", "invalid", True)]
assert "finite" in errors[0].message
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    fn row_validation_keeps_missing_provider_nonrejecting(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(c_str!(r#"
assert native.model_row_errors("model", {"litellm_provider": "openai", "input_cost_per_token": 0}) == []
errors = native.model_row_errors("team/model~", {"input_cost_per_token": 0})
assert [(e.key, e.field, e.kind, e.rejects) for e in errors] == [
    ("team/model~", None, "missing_provider", False)]
cyclic = {}
cyclic["self"] = cyclic
errors = native.model_row_errors("model", cyclic)
assert [(e.key, e.field, e.kind, e.rejects) for e in errors] == [("model", None, "invalid", True)]
assert "JSON-compatible" in errors[0].message
"#), Some(scope), Some(scope)).unwrap();
        });
    }

    #[rstest]
    #[case::absent(None, None, Some(json!({"litellm_provider":"openai", "input_cost_per_token":2, "cache_read_input_token_cost":4})))]
    #[case::insert(None, Some(json!({"input_cost_per_token":7})), Some(json!({"litellm_provider":"openai", "input_cost_per_token":7, "cache_read_input_token_cost":4})))]
    #[case::delete(Some(json!({"input_cost_per_token":1})), None, None)]
    #[case::edit(Some(json!({"input_cost_per_token":1, "cache_read_input_token_cost":3})), Some(json!({"input_cost_per_token":7})), Some(json!({"litellm_provider":"openai", "input_cost_per_token":7})))]
    #[case::equal(Some(json!({"input_cost_per_token":1})), Some(json!({"input_cost_per_token":1})), Some(json!({"litellm_provider":"openai", "input_cost_per_token":2, "cache_read_input_token_cost":4})))]
    #[case::nested(Some(json!({"metadata":{"keep":1,"remove":2}})), Some(json!({"metadata":{"keep":1}})), Some(json!({"litellm_provider":"openai", "input_cost_per_token":2, "cache_read_input_token_cost":4,"metadata":{"keep":1}})))]
    fn diff_composes_sparse_changes_onto_new_base(
        scope: Py<PyDict>,
        #[case] installed: Option<Value>,
        #[case] live: Option<Value>,
        #[case] expected: Option<Value>,
    ) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            scope
                .set_item("installed", to_py(py, &installed).unwrap())
                .unwrap();
            scope.set_item("live", to_py(py, &live).unwrap()).unwrap();
            scope
                .set_item("expected", to_py(py, &expected).unwrap())
                .unwrap();
            py.run(c_str!(r#"
base = native.ModelRegistry.parse(b'{"model":{"litellm_provider":"openai","input_cost_per_token":2,"cache_read_input_token_cost":4}}')
layers = native.ModelCostLayers()
layers.diff({"model": (installed, live)})
assert layers.compose_keys(base, frozenset({"model", "missing"})) == [
    ("missing", None, frozenset()), ("model", expected, frozenset())]
assert layers.compose(base) == ([] if expected is None else [("model", expected, frozenset())])
layers.forget_writes()
assert layers.compose(base) == [("model", base.served()["model"], frozenset())]
"#), Some(scope), Some(scope)).unwrap();
        });
    }

    #[rstest]
    fn registrations_keep_inheritance_and_direct_ownership_separate(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(c_str!(r#"
base = native.ModelRegistry.parse(b'{"model":{"litellm_provider":"openai","input_cost_per_token":2,"cache_read_input_token_cost":4}}')
layers = native.ModelCostLayers()
layers.record_registration("openai/model", "model", {"input_cost_per_token":5, "max_tokens":100})
registered = {"litellm_provider":"openai", "input_cost_per_token":5, "cache_read_input_token_cost":4, "max_tokens":100}
assert layers.compose_keys(base, frozenset({"openai/model"})) == [
    ("openai/model", registered, frozenset({"input_cost_per_token", "max_tokens"}))]
live = dict(registered, input_cost_per_token=7)
del live["cache_read_input_token_cost"]
layers.diff({"openai/model": (registered, live), "z": (None, {"extra":None}), "a": (None, {})})
assert layers.compose(base) == [
    ("model", base.served()["model"], frozenset()),
    ("openai/model", live, frozenset({"max_tokens"})),
    ("a", {}, frozenset()), ("z", {"extra":None}, frozenset()),
]
layers.record_registration("openai/model", "model", {"input_cost_per_token":9})
assert layers.compose_keys(base, frozenset({"openai/model"})) == [
    ("openai/model", dict(live, input_cost_per_token=9), frozenset({"max_tokens", "input_cost_per_token"}))]
layers.diff({"openai/model": (live, None)})
assert layers.compose_keys(base, frozenset({"openai/model"})) == [("openai/model", None, frozenset())]
"#), Some(scope), Some(scope)).unwrap();
        });
    }

    #[rstest]
    fn composition_shares_only_unchanged_alias_projections(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(c_str!(r#"
base = native.ModelRegistry.parse(b'{"model":{"litellm_provider":"openai","aliases":["alias"],"input_cost_per_token":2}}')
layers = native.ModelCostLayers()
initial = {key: row for key,row,_ in layers.compose(base)}
assert initial["model"] is initial["alias"]
layers.diff({"MODEL": (None, initial["model"])})
distinct = {key: row for key,row,_ in layers.compose(base)}
assert distinct["MODEL"] == distinct["model"]
assert distinct["MODEL"] is not distinct["model"]
layers.record_registration("alias", None, {"input_cost_per_token":2})
registered = {key: row for key,row,_ in layers.compose(base)}
assert registered["model"] == registered["alias"]
assert registered["model"] is not registered["alias"]
layers.diff({"alias": (initial["alias"], dict(initial["alias"], input_cost_per_token=3))})
assert layers.compose_keys(base, frozenset({"model", "alias"})) == [
    ("alias", {"litellm_provider":"openai", "input_cost_per_token":3}, frozenset()),
    ("model", initial["model"], frozenset()),
]
"#), Some(scope), Some(scope)).unwrap();
        });
    }

    #[rstest]
    fn write_rows_bypasses_subclass_hooks_and_delays_finalizers(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(
                c_str!(
                    r#"
class LiveDict(dict):
    def __setitem__(self, key, value):
        raise AssertionError("Python set hook called")
    def __delitem__(self, key):
        raise AssertionError("Python delete hook called")
    def clear(self):
        raise AssertionError("clear called")
observed = []
class Row(dict):
    def __del__(self):
        observed.append(dict(target))
target = LiveDict({"replaced": Row(), "removed": Row(), "untouched": {}})
untouched = target["untouched"]
new = {"replaced": {"rate":0}, "new": {}}
assert native.write_rows(target, new, ["removed", "missing"]) is None
expected = dict(new, untouched={})
assert type(target) is LiveDict
assert target == expected
assert target["untouched"] is untouched
assert target["replaced"] is new["replaced"]
assert observed == [expected, expected]
native.write_rows(target, target, ["new"])
assert target == {"untouched":{}, "replaced":{"rate":0}}
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    fn write_rows_is_atomic_to_python_observers(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(
                c_str!(
                    r#"
import sys
import threading
target = {f"old-{i}":{} for i in range(50)}
updates = {f"new-{i}":{} for i in range(3999)} | {"new-key":{}}
remove = list(target)
observed_sizes = set()
observed_membership = set()
ready, finished, after = threading.Event(), threading.Event(), threading.Event()
def observe():
    while not finished.is_set():
        observed_sizes.add(len(target))
        observed_membership.add("new-key" in target)
        ready.set()
        if "new-key" in target:
            after.set()
thread = threading.Thread(target=observe, daemon=True)
interval = sys.getswitchinterval()
try:
    sys.setswitchinterval(0.000001)
    thread.start()
    assert ready.wait(5)
    native.write_rows(target, updates, remove)
    assert after.wait(5)
finally:
    finished.set()
    thread.join(5)
    sys.setswitchinterval(interval)
assert not thread.is_alive()
assert target == updates
assert observed_sizes == {50, 4000}, observed_sizes
assert observed_membership == {False, True}
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }

    #[rstest]
    fn compose_releases_gil_for_a_python_thread(scope: Py<PyDict>) {
        Python::attach(|py| {
            let scope = scope.bind(py);
            py.run(
                c_str!(
                    r#"
import sys
import threading
base = native.ModelRegistry.bundled()
layers = native.ModelCostLayers()
ready, begin, ran = threading.Event(), threading.Event(), threading.Event()
def observer():
    ready.set()
    begin.wait()
    ran.set()
thread = threading.Thread(target=observer, daemon=True)
thread.start()
assert ready.wait(5)
interval = sys.getswitchinterval()
try:
    sys.setswitchinterval(60)
    begin.set()
    rows = layers.compose(base)
    ran_during_compose = ran.is_set()
finally:
    sys.setswitchinterval(interval)
    thread.join(5)
assert ran_during_compose, "compose held the GIL"
assert {key:row for key,row,_ in rows} == base.served()
"#
                ),
                Some(scope),
                Some(scope),
            )
            .unwrap();
        });
    }
}
