mod install;
mod row_error;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use litellm_host_python::{release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, IntegrityLimits, Layers};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyDict, PyModule};
use serde::Serialize;
use sha1::{Digest, Sha1};

use install::{Record, write_into};
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

struct Parsed {
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
}

struct State {
    layers: Layers,
    record: Option<Record>,
}

#[pyclass(frozen, name = "Catalog", module = "litellm.rust_bridge._native")]
pub(crate) struct CatalogHandle {
    module: Py<PyModule>,
    attr: String,
    state: Mutex<State>,
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
impl CatalogHandle {
    #[staticmethod]
    fn bound_to(module: Bound<'_, PyModule>, attr: String) -> Self {
        Self {
            module: module.unbind(),
            attr,
            state: Mutex::new(State {
                layers: Layers::new(Arc::clone(&BUNDLED.catalog)),
                record: None,
            }),
        }
    }

    fn bundled(&self) -> Fetched {
        Fetched {
            parsed: Arc::clone(&BUNDLED),
            revision: BUNDLED_REVISION.clone(),
            etag: None,
        }
    }

    /// Parses a fetched body and applies main's whole-map checks against the bundled model count
    #[staticmethod]
    #[pyo3(signature = (body, *, etag, min_model_count, max_shrink_ratio))]
    fn parse(
        py: Python<'_>,
        body: PyBackedBytes,
        etag: Option<String>,
        min_model_count: usize,
        max_shrink_ratio: f64,
    ) -> PyResult<Py<PyAny>> {
        let parsed = release_gil(py, || {
            parse_remote(&body, min_model_count, max_shrink_ratio)
        });
        match parsed {
            Ok((parsed, revision)) => Ok(Fetched {
                parsed: Arc::new(parsed),
                revision,
                etag,
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
            Err(reason) => Ok(Unavailable { reason }
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    /// Writes the composed rows into the dict bound to the attribute, never over a field the user owns (decision 25)
    fn install(&self, py: Python<'_>, fetched: &Fetched) -> PyResult<Installed> {
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
        self.lock().record = Some(outcome.record);
        Ok(Installed {
            count: composed.len(),
            written: outcome.written,
            removed: outcome.removed,
            rejected: fetched.parsed.rejected.clone(),
        })
    }
}

impl CatalogHandle {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn parse_remote(
    body: &[u8],
    min_model_count: usize,
    max_shrink_ratio: f64,
) -> Result<(Parsed, String), String> {
    let limits = IntegrityLimits::new(min_model_count, max_shrink_ratio)
        .ok_or_else(|| format!("invalid shrink ratio {max_shrink_ratio}"))?;
    let loaded = Catalog::parse(body).map_err(|error| error.to_string())?;
    loaded
        .value
        .check_size(BUNDLED.raw.model_count(), limits)
        .map_err(|error| error.to_string())?;
    let raw = Arc::new(loaded.value.clone());
    let errors = loaded.errors.iter().map(row_error).collect();
    let (catalog, rejections) = loaded
        .value
        .with_rejected_replaced(&loaded.errors, &BUNDLED.raw);
    let rejected = rejections
        .iter()
        .flat_map(|rejection| rejection.errors.iter().copied().map(row_error))
        .collect();
    Ok((
        Parsed {
            catalog: Arc::new(catalog),
            raw,
            errors,
            rejected,
        },
        git_blob_id(body),
    ))
}

fn git_blob_id(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn the_revision_is_the_git_blob_id_of_the_body() {
        assert_eq!(
            git_blob_id(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }
}
