mod fetch;
mod install;
mod row_error;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::SystemTime;

use litellm_host_python::{release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, Error as CatalogError, Installs};
use litellm_model_catalog_refresh::{Source, git_blob_id};
use parking_lot::Mutex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyModule};
use serde::Serialize;

pub(crate) use fetch::Provenance;
use install::BoundDict;
pub(crate) use row_error::{RowError, row_error};

static BUNDLED: LazyLock<Arc<Contents>> = LazyLock::new(|| {
    let loaded = Catalog::parse(BUNDLED_REGISTRY).expect("the bundled model registry parses");
    let catalog = Arc::new(loaded.value);
    Arc::new(Contents {
        raw: Arc::clone(&catalog),
        catalog,
        errors: loaded.errors.iter().map(row_error).collect(),
        rejected: Vec::new(),
    })
});

static BUNDLED_REVISION: LazyLock<String> = LazyLock::new(|| git_blob_id(BUNDLED_REGISTRY));

pub(crate) struct Contents {
    /// What install layers over bundled, rejected rows replaced or left out
    catalog: Arc<Catalog>,
    /// The document as written
    raw: Arc<Catalog>,
    errors: Vec<RowError>,
    rejected: Vec<RowError>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Parsed {
    contents: Arc<Contents>,
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
    origin: MapOrigin,
    /// When the load that produced this map started, main's `loaded_at`
    started_at: SystemTime,
}

/// Where a `Parsed` came from, which install records as its provenance
#[derive(Clone)]
pub(crate) enum MapOrigin {
    Bundled,
    /// `LITELLM_LOCAL_MODEL_COST_MAP` or `LITELLM_OFFLINE_MODE` chose the bundled map
    Forced,
    /// The bundled map served because loading `url` failed
    Fallback {
        url: String,
        reason: String,
    },
    Remote(Source),
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
    /// Fetched keys rejected by validation, empty for the bundled map
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
    /// The URL that failed, `None` when the failure came from no load
    url: Option<String>,
    started_at: SystemTime,
}

struct State {
    installs: Installs,
    provenance: Option<Provenance>,
}

#[pyclass(frozen, name = "Catalog", module = "litellm.rust_bridge._native")]
pub(crate) struct CatalogHandle {
    module: Py<PyModule>,
    attr: String,
    state: Mutex<State>,
}

#[pymethods]
impl Parsed {
    /// The rows of this one document as install would serve them, without layers
    fn served(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let served = release_gil(py, || served_rows(&self.contents.catalog));
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
        to_py(py, &WrittenDocument(self.contents.raw.as_ref()))
    }

    fn fallback_rules(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &self.contents.raw.fallback_rules())
    }

    fn errors(&self) -> Vec<RowError> {
        self.contents.errors.clone()
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
            url: None,
            started_at: SystemTime::now(),
        }
    }

    /// The bundled map, recording this failure as the reason it is served
    fn fallback(&self) -> Parsed {
        Parsed {
            origin: MapOrigin::Fallback {
                url: self.url.clone().unwrap_or_default(),
                reason: format!("Remote fetch failed: {}", self.reason),
            },
            started_at: self.started_at,
            ..bundled()
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
                installs: Installs::new(Arc::clone(&BUNDLED.catalog)),
                provenance: None,
            }),
        }
    }

    fn bundled(&self) -> Parsed {
        bundled()
    }

    /// The map the operator configured for `url`, installing nothing. See `fetch::load`
    #[pyo3(signature = (url, *, timeout, attempts))]
    fn load(
        &self,
        py: Python<'_>,
        url: String,
        timeout: f64,
        attempts: u32,
    ) -> PyResult<Py<PyAny>> {
        fetch::load(py, url, timeout, attempts)
    }

    /// Where the installed map came from, `None` before the first install
    fn provenance(&self) -> Option<Provenance> {
        self.state.lock().provenance.clone()
    }

    /// Writes the composed rows into the dict bound to the attribute, never over a field the user owns, and records
    /// where `parsed` came from
    fn install(&self, py: Python<'_>, parsed: &Parsed) -> PyResult<Installed> {
        let bound = self.module.bind(py).getattr(self.attr.as_str())?;
        let dict = bound
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("{} must be a dict", self.attr)))?;
        let dict_id = dict.as_ptr() as usize;
        let pending = self
            .state
            .lock()
            .installs
            .begin(Arc::clone(&parsed.contents.catalog), dict_id);
        let composed = release_gil(py, || pending.compose());
        let outcome = pending.write(&mut BoundDict::new(dict), &composed)?;
        let provenance = Provenance::of(parsed);
        {
            let mut state = self.state.lock();
            state.installs.finish(outcome.record);
            state.provenance = Some(provenance);
        }
        Ok(Installed {
            count: composed.len(),
            written: outcome.written,
            removed: outcome.removed,
            rejected: parsed.contents.rejected.clone(),
        })
    }
}

pub(crate) fn bundled() -> Parsed {
    Parsed {
        contents: Arc::clone(&BUNDLED),
        revision: BUNDLED_REVISION.clone(),
        etag: None,
        origin: MapOrigin::Bundled,
        started_at: SystemTime::now(),
    }
}

/// A fetched document with the rejected rows replaced by bundled or left out
pub(crate) fn remote(catalog: Catalog, errors: &[CatalogError]) -> Contents {
    let raw = Arc::new(catalog.clone());
    let (catalog, rejections) = catalog.with_rejected_replaced(errors, &BUNDLED.raw);
    Contents {
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

#[pyfunction]
pub fn bundled_model_cost_map(py: Python<'_>) -> Bound<'_, PyBytes> {
    static BYTES: PyOnceLock<Py<PyBytes>> = PyOnceLock::new();
    BYTES
        .get_or_init(py, || {
            PyBytes::new(py, litellm_model_catalog::BUNDLED_REGISTRY).unbind()
        })
        .bind(py)
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn repeated_calls_reuse_the_bundled_registry_bytes() {
        Python::initialize();
        let first = Python::attach(|py| bundled_model_cost_map(py).unbind());
        Python::attach(|py| {
            let second = bundled_model_cost_map(py);
            assert!(first.bind(py).is(&second));
            assert_eq!(second.as_bytes(), litellm_model_catalog::BUNDLED_REGISTRY);
        });
    }
}
