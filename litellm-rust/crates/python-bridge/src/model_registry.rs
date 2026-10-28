mod fetch;
mod install;
mod name_index;
mod py_values;
mod refresher;
mod register;
mod resolve;
mod row_error;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::SystemTime;

use litellm_host_python::{release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, Keys, Layers, Located};
use litellm_model_catalog_refresh::{Source, git_blob_id};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde::Serialize;

pub(crate) use fetch::{Bundled, File, Provenance, Remote};
use install::{Record, write_into};
pub(crate) use refresher::Refresher;
pub(crate) use register::Registered;
pub(crate) use resolve::{Invalid, Missing, Resolved};
pub(crate) use row_error::{RowError, RowErrorKind, row_error};

static BUNDLED: LazyLock<Arc<Parsed>> = LazyLock::new(|| {
    let loaded = Catalog::parse(BUNDLED_REGISTRY).expect("the bundled model registry parses");
    let catalog = Arc::new(loaded.value);
    Arc::new(Parsed {
        raw: Arc::clone(&catalog),
        catalog,
        errors: loaded.errors.iter().map(row_error).collect(),
        rejected: Vec::new(),
    })
});

static BUNDLED_REVISION: LazyLock<String> = LazyLock::new(|| git_blob_id(BUNDLED_REGISTRY));

pub(crate) struct Parsed {
    /// What install layers over bundled, rows decision 15 rejects replaced or left out
    catalog: Arc<Catalog>,
    /// The document as written
    raw: Arc<Catalog>,
    errors: Vec<RowError>,
    rejected: Vec<RowError>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Fetched {
    parsed: Arc<Parsed>,
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
    /// `None` for the bundled map
    source: Option<Source>,
    /// When the load that produced this map started, main's `loaded_at`
    started_at: SystemTime,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Installed {
    /// Rows the composed map serves, what main's adopt returns as the fetched map's length
    #[pyo3(get)]
    count: usize,
    /// Keys set because the row differed or was new
    #[pyo3(get)]
    written: usize,
    /// Keys deleted because the map no longer serves them
    #[pyo3(get)]
    removed: usize,
    /// Decision 15, empty for the bundled map
    #[pyo3(get)]
    rejected: Vec<RowError>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Unavailable {
    #[pyo3(get)]
    reason: String,
    /// Seconds to wait before retrying, set when the last reply was retryable
    #[pyo3(get)]
    retry_after: Option<f64>,
}

struct State {
    layers: Layers,
    record: Option<Record>,
    provenance: Option<Provenance>,
}

#[pyclass(frozen, name = "Catalog", module = "litellm.rust_bridge._native")]
pub(crate) struct CatalogHandle {
    module: Py<PyModule>,
    attr: String,
    state: Mutex<State>,
    index: Mutex<Option<name_index::NameIndex>>,
    /// Bumped by every install, so a registration that raced one recomputes its base
    generation: AtomicU64,
}

#[pymethods]
impl Fetched {
    /// The rows of this one document as install would serve them, without layers
    fn served(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let served = release_gil(py, || served_rows(&self.parsed.catalog));
        let rows = PyDict::new(py);
        let mut shared: HashMap<String, Py<PyDict>> = HashMap::new();
        for (key, owner, fields) in served {
            let row = match shared.get(&owner) {
                Some(row) => row.clone_ref(py),
                None => {
                    let row = fields_to_dict(py, &fields)?;
                    shared.insert(owner, row.clone_ref(py));
                    row
                }
            };
            rows.set_item(key, row)?;
        }
        Ok(rows.unbind())
    }

    /// The raw document, `sample_spec` and `fallback_generalizations` included
    fn document(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &WrittenDocument(self.parsed.raw.as_ref()))
    }

    fn fallback_rules(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &self.parsed.raw.fallback_rules())
    }

    fn errors(&self) -> Vec<RowError> {
        self.parsed.errors.clone()
    }
}

/// The merge rule for `register_model`, and with `replace_variants` the one for `model_info_with_pricing_overrides`,
/// where a flat price also replaces the tier, batch, off-peak and reasoning rates it stands in for
#[pyfunction]
#[pyo3(signature = (base, registration, *, replace_variants = false))]
pub(crate) fn overlay(
    py: Python<'_>,
    base: &Bound<'_, PyDict>,
    registration: &Bound<'_, PyDict>,
    replace_variants: bool,
) -> PyResult<Py<PyAny>> {
    let base: Fields = litellm_host_python::from_py(base)?;
    let registration: Fields = litellm_host_python::from_py(registration)?;
    let merged = if replace_variants {
        litellm_model_catalog::update_dictionary(
            &litellm_model_catalog::without_overridden_price_variants(&base, &registration),
            &registration,
        )
    } else {
        litellm_model_catalog::overlay(&base, &registration)
    };
    to_py(py, &merged)
}

#[pymethods]
impl Unavailable {
    #[new]
    #[pyo3(signature = (reason, retry_after = None))]
    fn new(reason: String, retry_after: Option<f64>) -> Self {
        Self {
            reason,
            retry_after,
        }
    }
}

#[pymethods]
impl CatalogHandle {
    #[staticmethod]
    fn bound_to(module: Bound<'_, PyModule>, attr: String) -> Self {
        Self {
            module: module.unbind(),
            attr,
            state: Mutex::new(State {
                layers: Layers::new(Arc::clone(&BUNDLED.catalog)),
                record: None,
                provenance: None,
            }),
            index: Mutex::new(None),
            generation: AtomicU64::new(0),
        }
    }

    fn bundled(&self) -> Fetched {
        bundled()
    }

    /// Fetches and validates `source` with no install, retrying up to `attempts` times
    #[pyo3(signature = (source, *, timeout, attempts))]
    fn fetch(
        &self,
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        timeout: f64,
        attempts: u32,
    ) -> PyResult<Py<PyAny>> {
        fetch::fetch(py, fetch::requested(source)?, timeout, attempts)
    }

    /// A background refresh of the bound map from `source`, built and not started (decisions 2 and 4)
    fn refresher(slf: Py<Self>, py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Refresher> {
        match fetch::requested(source)? {
            fetch::Requested::Source(source) => Ok(Refresher::new(slf.clone_ref(py), source)),
            fetch::Requested::Bundled => Err(pyo3::exceptions::PyValueError::new_err(
                "the bundled map needs no refresh",
            )),
            fetch::Requested::Invalid(reason) => {
                Err(pyo3::exceptions::PyValueError::new_err(reason))
            }
        }
    }

    /// Fetch with main's reload policy, 3 attempts of 5 s, then install
    fn reload(&self, py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let requested = fetch::requested(source)?;
        let env_forced = matches!(requested, fetch::Requested::Bundled);
        let fetched = fetch::fetch(py, requested, 5.0, 3)?;
        let Ok(fetched) = fetched.bind(py).cast::<Fetched>() else {
            return Ok(fetched);
        };
        Ok(self
            .install(py, fetched.get(), env_forced, None, None)?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    /// The bound dict's row for `model`, through main's candidate names (decisions 18 and 20)
    #[pyo3(signature = (model, provider = None))]
    fn resolve(&self, py: Python<'_>, model: &str, provider: Option<&str>) -> PyResult<Py<PyAny>> {
        let dict = self.bound_dict(py)?;
        let keys = resolve::LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        resolve::resolve(py, &keys, model, provider)
    }

    /// The actual key for `name`, exact first, then through the case-insensitive index
    fn lookup_key(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        let dict = self.bound_dict(py)?;
        let keys = resolve::LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        Ok(if keys.contains(name) {
            Some(name.to_owned())
        } else {
            keys.by_normalized(name)
        })
    }

    /// The row for an image call, through main's size and quality key spellings
    #[pyo3(signature = (model, provider = None, *, size = None, quality = None))]
    fn resolve_image(
        &self,
        py: Python<'_>,
        model: &str,
        provider: Option<&str>,
        size: Option<(u32, u32)>,
        quality: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let dict = self.bound_dict(py)?;
        let keys = resolve::LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        let candidates = litellm_model_catalog::ImageCandidates {
            model: model.to_owned(),
            provider: provider.map(str::to_owned),
            size,
            quality: quality.map(str::to_owned),
        };
        resolve::resolve_image(py, &keys, &candidates)
    }

    /// fal's nearest size row, `{provider}/{quality}/{W}-x-{H}/{model}` with an `output_cost_per_image` price, from the
    /// provider's keys in the index
    #[pyo3(signature = (model, provider, *, quality, size))]
    fn nearest_image_key(
        &self,
        py: Python<'_>,
        model: &str,
        provider: &str,
        quality: &str,
        size: (u32, u32),
    ) -> PyResult<Option<String>> {
        let dict = self.bound_dict(py)?;
        let keys = resolve::LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        let priced: Vec<String> = keys
            .provider_keys(provider)
            .into_iter()
            .filter(|key| {
                keys.fields_of(key)
                    .and_then(|fields| {
                        fields
                            .get("output_cost_per_image")
                            .map(serde_json::Value::is_number)
                    })
                    .unwrap_or(false)
            })
            .collect();
        Ok(litellm_model_catalog::nearest_image_key(
            priced.iter().map(String::as_str),
            provider,
            model,
            quality,
            size,
        )
        .map(str::to_owned))
    }

    /// The keys whose row names this `litellm_provider`, from the index
    fn provider_keys(&self, py: Python<'_>, provider: &str) -> PyResult<Vec<String>> {
        let dict = self.bound_dict(py)?;
        let keys = resolve::LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        Ok(keys.provider_keys(provider))
    }

    /// main's 133-field copy of a row Python built, a generalization match or a fill-missing merge
    #[staticmethod]
    #[pyo3(signature = (key, row, provider = None))]
    fn model_info_of(
        py: Python<'_>,
        key: &str,
        row: &Bound<'_, PyDict>,
        provider: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let fields: Fields = row
            .iter()
            .filter_map(|(name, value)| {
                Some((
                    name.extract::<String>().ok()?,
                    litellm_host_python::from_py::<serde_json::Value>(&value).ok()?,
                ))
            })
            .collect();
        to_py(
            py,
            &litellm_model_catalog::model_info(key, &fields, provider),
        )
    }

    /// Writes each row under its caller's key on top of the row it inherits from, into layer 3 when `persist`
    #[pyo3(signature = (rows, *, persist, provider = None))]
    fn register(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyDict>,
        persist: bool,
        provider: Option<&str>,
    ) -> PyResult<Registered> {
        self.register_rows(py, rows, persist, provider)
    }

    /// The row a registration of `key` inherits from, or the capability seed
    #[pyo3(signature = (key, provider = None))]
    fn registration_base(
        &self,
        py: Python<'_>,
        key: &str,
        provider: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        to_py(py, &self.base_for(py, key, provider)?)
    }

    /// main's replay after an install, for a caller that swapped the dict itself
    fn replay_registrations(&self, py: Python<'_>) -> PyResult<()> {
        self.replay_after_install(py)
    }

    /// The fields any registration set on `key` since the last install
    fn registered_fields(&self, key: &str) -> std::collections::BTreeSet<String> {
        self.lock()
            .layers
            .registrations()
            .registered_fields(key)
            .cloned()
            .unwrap_or_default()
    }

    /// The persisted registrations, a snapshot
    fn registrations(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let persisted = self.lock().layers.registrations().persisted().clone();
        to_py(py, &persisted)
    }

    /// Test isolation only, puts back a `registrations()` snapshot
    fn restore_registrations(&self, snapshot: &Bound<'_, PyDict>) -> PyResult<()> {
        let persisted: indexmap::IndexMap<String, Fields> = litellm_host_python::from_py(snapshot)?;
        self.lock().layers.registrations_mut().restore(persisted);
        Ok(())
    }

    /// Rebuild the index on the next lookup, for a caller that knows keys changed with the length unchanged
    fn mark_index_stale(&self) {
        *self.index.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// Where the installed map came from, `None` before the first install
    fn provenance(&self) -> Option<Provenance> {
        self.lock().provenance.clone()
    }

    /// Writes the composed rows into the dict bound to the attribute, never over a field the user owns (decision 25).
    /// `env_forced`, `fallback_reason` and `url` are the provenance facts only the caller knows
    #[pyo3(signature = (fetched, *, env_forced = false, fallback_reason = None, url = None))]
    fn install(
        &self,
        py: Python<'_>,
        fetched: &Fetched,
        env_forced: bool,
        fallback_reason: Option<String>,
        url: Option<String>,
    ) -> PyResult<Installed> {
        let bound = self.module.bind(py).getattr(self.attr.as_str())?;
        let dict = bound
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("{} must be a dict", self.attr)))?;
        let dict_id = dict.as_ptr() as usize;
        let (layers, record) = {
            let mut state = self.lock();
            state
                .layers
                .replace_fetched(Arc::clone(&fetched.parsed.catalog));
            let record = state
                .record
                .clone()
                .filter(|record| record.dict_id == dict_id);
            (state.layers.clone(), record)
        };
        let composed = release_gil(py, || layers.compose());
        let outcome = write_into(py, dict, &composed, &layers, record)?;
        let provenance = Provenance::of(fetched, env_forced, fallback_reason, url);
        {
            let mut state = self.lock();
            state.record = Some(outcome.record);
            state.provenance = Some(provenance);
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.replay_after_install(py)?;
        Ok(Installed {
            count: composed.len(),
            written: outcome.written,
            removed: outcome.removed,
            rejected: fetched.parsed.rejected.clone(),
        })
    }
}

impl CatalogHandle {
    pub(crate) fn install_generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn bound_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let bound = self.module.bind(py).getattr(self.attr.as_str())?;
        bound
            .cast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("{} must be a dict", self.attr)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) fn bundled() -> Fetched {
    Fetched {
        parsed: Arc::clone(&BUNDLED),
        revision: BUNDLED_REVISION.clone(),
        etag: None,
        source: None,
        started_at: SystemTime::now(),
    }
}

/// A fetched document with the rows decision 15 rejects replaced by bundled or left out
pub(crate) fn remote(catalog: Catalog, errors: &[Located]) -> Parsed {
    let raw = Arc::new(catalog.clone());
    let (catalog, rejections) = catalog.with_rejected_replaced(errors, &BUNDLED.raw);
    Parsed {
        catalog: Arc::new(catalog),
        raw,
        errors: errors.iter().map(row_error).collect(),
        rejected: rejections
            .iter()
            .flat_map(|rejection| rejection.errors.iter().copied().map(row_error))
            .collect(),
    }
}

fn served_rows(catalog: &Catalog) -> Vec<(String, String, Fields)> {
    catalog
        .served()
        .map(|(key, fields)| {
            let owner = catalog
                .aliases()
                .get(key)
                .cloned()
                .unwrap_or_else(|| key.to_owned());
            let fields = fields
                .iter()
                .filter(|(field, _)| *field != "aliases")
                .map(|(field, value)| (field.clone(), value.clone()))
                .collect();
            (key.to_owned(), owner, fields)
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

struct WrittenDocument<'a>(&'a Catalog);

impl Serialize for WrittenDocument<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.written())
    }
}
