use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use litellm_host_python::{from_py, release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    ApplicationWrite, BUNDLED_REGISTRY, Catalog, Error, IntegrityLimits, Layers, Located, Owner,
    Precedence, RegistrationOptions, SparseRow,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyBytes, PyDict, PyFrozenSet, PyList, PyTuple};
use serde::Serialize;
use strum::{IntoStaticStr, VariantArray};

mod transport;
pub(crate) use transport::{
    AdoptStep, DoneStep, FetchStep, FetchedRegistry, RegistryNotModified, RegistryUnavailable,
    RevalidationState, SleepStep, fetch_model_registry, fetch_model_registry_async, revalidation,
};

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
        })
    }

    #[staticmethod]
    fn bundled() -> Self {
        Self {
            parsed: Arc::new(ParsedRegistry {
                catalog: Arc::clone(&BUNDLED.catalog),
                errors: Arc::clone(&BUNDLED.errors),
            }),
        }
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

    #[pyo3(signature = (reference=None, minimum=None, ratio=None))]
    fn check_size(
        &self,
        reference: Option<usize>,
        minimum: Option<usize>,
        ratio: Option<f64>,
    ) -> bool {
        let policy = litellm_model_catalog::RetryPolicy::default();
        let reference = reference.unwrap_or_else(|| BUNDLED.catalog.model_count());
        let minimum = minimum.unwrap_or(policy.min_model_count);
        let ratio = ratio.unwrap_or(policy.min_reference_ratio);
        IntegrityLimits::new(minimum, ratio)
            .is_some_and(|limits| self.parsed.catalog.check_size(reference, limits).is_ok())
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

fn required_registration_option<'py>(
    options: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    options
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("registration option {name} is required")))
}

fn registration_options_from_dict(options: &Bound<'_, PyDict>) -> PyResult<RegistrationOptions> {
    Ok(RegistrationOptions {
        alias_keys: required_registration_option(options, "alias_keys")?.extract()?,
        defaults: from_py(&required_registration_option(options, "defaults")?)?,
        reset_fields: from_py(&required_registration_option(options, "reset_fields")?)?,
        cleanup_overrides: required_registration_option(options, "cleanup_overrides")?.extract()?,
        fallback: required_registration_option(options, "fallback")?.extract()?,
        names_the_row: required_registration_option(options, "names_the_row")?.extract()?,
        ignore_none: required_registration_option(options, "ignore_none")?.extract()?,
    })
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
        key: &str,
        inherits_from: Option<String>,
        fields: &Bound<'_, PyDict>,
    ) -> PyResult<u64> {
        let fields: SparseRow = from_py(fields)?;
        Ok(self.lock().record_registration(
            key.to_owned(),
            inherits_from,
            fields,
            RegistrationOptions {
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        ))
    }

    #[pyo3(signature = (key, inherits_from, fields, options))]
    fn record_registration_with_options(
        &self,
        key: &str,
        inherits_from: Option<String>,
        fields: &Bound<'_, PyDict>,
        options: &Bound<'_, PyDict>,
    ) -> PyResult<u64> {
        Ok(self.lock().record_registration(
            key.to_owned(),
            inherits_from,
            from_py(fields)?,
            registration_options_from_dict(options)?,
        ))
    }

    #[pyo3(signature = (handle, key, inherits_from, fields, options))]
    fn rebind_registration(
        &self,
        handle: u64,
        key: &str,
        inherits_from: Option<String>,
        fields: &Bound<'_, PyDict>,
        options: &Bound<'_, PyDict>,
    ) -> PyResult<bool> {
        Ok(self.lock().rebind_registration(
            handle,
            key.to_owned(),
            inherits_from,
            from_py(fields)?,
            registration_options_from_dict(options)?,
        ))
    }

    fn record_direct(&self, key: &str, row: &Bound<'_, PyDict>) -> PyResult<()> {
        let row: Fields = from_py(row)?;
        self.lock().record(ApplicationWrite::Direct {
            key: key.to_owned(),
            row,
        });
        Ok(())
    }

    fn record_tombstone(&self, key: &str) {
        self.lock().record(ApplicationWrite::Tombstone {
            key: key.to_owned(),
        });
    }

    fn set_owner(&self, name: &str, precedence: &str, rows: &Bound<'_, PyDict>) -> PyResult<()> {
        let rows = from_py(rows)?;
        let precedence = match precedence {
            "fallback" => Precedence::Fallback,
            "override" => Precedence::Override,
            _ => {
                return Err(PyValueError::new_err(
                    "precedence must be fallback or override",
                ));
            }
        };
        self.lock().set_owner(Owner {
            name: name.to_owned(),
            precedence,
            rows,
        });
        Ok(())
    }

    fn drop_owner(&self, name: &str) {
        self.lock().drop_owner(name);
    }

    fn forget_writes(&self) {
        self.lock().forget_writes();
    }

    fn snapshot(&self) -> Self {
        Self {
            layers: Mutex::new(self.lock().clone()),
        }
    }

    fn restore(&self, snapshot: &Self) {
        let restored = snapshot.lock().clone();
        *self.lock() = restored;
    }

    fn compose(&self, py: Python<'_>, base: &ModelRegistry) -> PyResult<Py<PyAny>> {
        let catalog = Arc::clone(&base.parsed.catalog);
        let rows = release_gil(py, || composed_rows(&self.lock(), catalog.as_ref()));
        composed_rows_to_py(py, rows)
    }

    fn compose_keys(
        &self,
        py: Python<'_>,
        base: &ModelRegistry,
        keys: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let keys: BTreeSet<String> = from_py(keys)?;
        let catalog = Arc::clone(&base.parsed.catalog);
        let rows = release_gil(py, || {
            self.lock()
                .compose_keys(catalog.as_ref(), &keys)
                .zip(keys.iter())
                .map(|(row, key)| {
                    (
                        key.clone(),
                        row.map(|row| (fields_without_aliases(&row.fields), row.registered)),
                    )
                })
                .collect::<Vec<_>>()
        });
        let items = PyList::empty(py);
        for (key, row) in rows {
            let key = key.into_pyobject(py)?.into_any();
            let (fields, registered) = match row {
                Some((fields, registered)) => (
                    fields_to_dict(py, &fields)?.into_bound(py).into_any(),
                    PyFrozenSet::new(py, registered.iter())?.into_any(),
                ),
                None => (py.None().into_bound(py), PyFrozenSet::empty(py)?.into_any()),
            };
            items.append(PyTuple::new(py, [key, fields, registered])?)?;
        }
        Ok(items.unbind().into_any())
    }

    fn compose_prefix(
        &self,
        py: Python<'_>,
        base: &ModelRegistry,
        handle: u64,
    ) -> PyResult<Py<PyAny>> {
        let catalog = Arc::clone(&base.parsed.catalog);
        let rows = release_gil(py, || {
            self.lock()
                .compose_prefix(catalog.as_ref(), handle)
                .map(|rows| {
                    rows.into_iter()
                        .map(|row| {
                            (
                                row.key.to_owned(),
                                row.group,
                                fields_without_aliases(&row.fields),
                                row.registered,
                            )
                        })
                        .collect::<Vec<_>>()
                })
        })
        .ok_or_else(|| PyValueError::new_err("registration handle is no longer present"))?;
        composed_rows_to_py(py, rows)
    }
}

impl ModelCostLayers {
    fn lock(&self) -> std::sync::MutexGuard<'_, Layers> {
        self.layers.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) fn fields_without_aliases(fields: &Fields) -> Fields {
    let mut fields = fields.clone();
    fields.remove("aliases");
    fields
}

fn canonical_groups(catalog: &Catalog) -> HashMap<String, String> {
    catalog
        .served()
        .map(|(key, _)| {
            (
                key.to_owned(),
                catalog.lookup(key).map_or_else(
                    || key.to_owned(),
                    |matched| matched.canonical_key.to_owned(),
                ),
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

type ComposedFields = (String, String, Fields, BTreeSet<String>);

struct ComposedRowProjection {
    fields: Fields,
    registered: BTreeSet<String>,
    row: Py<PyDict>,
}

fn composed_rows(layers: &Layers, catalog: &Catalog) -> Vec<ComposedFields> {
    layers
        .compose(catalog)
        .map(|row| {
            (
                row.key.to_owned(),
                row.group,
                fields_without_aliases(&row.fields),
                row.registered,
            )
        })
        .collect()
}

fn composed_rows_to_py(py: Python<'_>, rows: Vec<ComposedFields>) -> PyResult<Py<PyAny>> {
    let mut converted = HashMap::<String, Vec<ComposedRowProjection>>::new();
    let values = PyList::empty(py);
    for (key, group, fields, registered) in rows {
        let row = converted.get(&group).and_then(|projections| {
            projections.iter().find_map(|projection| {
                (projection.fields == fields && projection.registered == registered)
                    .then(|| projection.row.clone_ref(py))
            })
        });
        let row = match row {
            Some(row) => row,
            None => {
                let row = fields_to_dict(py, &fields)?;
                converted
                    .entry(group)
                    .or_default()
                    .push(ComposedRowProjection {
                        fields,
                        registered: registered.clone(),
                        row: row.clone_ref(py),
                    });
                row
            }
        };
        let name = key.into_pyobject(py)?.into_any();
        let fields = row.into_bound(py).into_any();
        let registered = PyFrozenSet::new(py, registered.iter())?.into_any();
        values.append(PyTuple::new(py, [name, fields, registered])?)?;
    }
    Ok(values.unbind().into_any())
}
