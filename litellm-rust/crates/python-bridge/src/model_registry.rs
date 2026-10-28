mod install;
mod row_error;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use litellm_host_python::{release_gil, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, Installs, IntegrityLimits};
use parking_lot::Mutex;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyModule};
use serde::Serialize;
use sha1::{Digest, Sha1};

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

struct Contents {
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
}

#[pyclass(frozen, name = "Catalog", module = "litellm.rust_bridge._native")]
pub(crate) struct CatalogHandle {
    module: Py<PyModule>,
    attr: String,
    installs: Mutex<Installs>,
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
impl CatalogHandle {
    #[staticmethod]
    fn bound_to(module: Bound<'_, PyModule>, attr: String) -> Self {
        Self {
            module: module.unbind(),
            attr,
            installs: Mutex::new(Installs::new(Arc::clone(&BUNDLED.catalog))),
        }
    }

    fn bundled(&self) -> Parsed {
        Parsed {
            contents: Arc::clone(&BUNDLED),
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
        let result = release_gil(py, || {
            parse_remote(&body, min_model_count, max_shrink_ratio)
        });
        match result {
            Ok((contents, revision)) => Ok(Parsed {
                contents: Arc::new(contents),
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

    /// Writes the composed rows into the dict bound to the attribute, never over a field the user owns
    fn install(&self, py: Python<'_>, parsed: &Parsed) -> PyResult<Installed> {
        let bound = self.module.bind(py).getattr(self.attr.as_str())?;
        let dict = bound
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("{} must be a dict", self.attr)))?;
        let dict_id = dict.as_ptr() as usize;
        let pending = self
            .installs
            .lock()
            .begin(Arc::clone(&parsed.contents.catalog), dict_id);
        let composed = release_gil(py, || pending.compose());
        let outcome = pending.write(&mut BoundDict::new(dict), &composed)?;
        self.installs.lock().finish(outcome.record);
        Ok(Installed {
            count: composed.len(),
            written: outcome.written,
            removed: outcome.removed,
            rejected: parsed.contents.rejected.clone(),
        })
    }
}

fn parse_remote(
    body: &[u8],
    min_model_count: usize,
    max_shrink_ratio: f64,
) -> Result<(Contents, String), String> {
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
        Contents {
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
mod revision_tests {
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
