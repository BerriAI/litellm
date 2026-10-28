mod fetch;
mod install;
mod name_index;
mod resolve;
mod row_error;

use std::collections::HashMap;
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
        Ok(Installed {
            count: composed.len(),
            written: outcome.written,
            removed: outcome.removed,
            rejected: fetched.parsed.rejected.clone(),
        })
    }
}

impl CatalogHandle {
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
