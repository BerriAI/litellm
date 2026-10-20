//! The Python face of `litellm_model_catalog::ServedCatalog`. Every method converts to or from
//! Python and delegates, so the registration semantics live and are tested in the catalog crate.
use std::sync::Arc;

use indexmap::IndexMap;
use litellm_host_python::{from_py_argument, release_gil, to_py};
use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, Fields, IntegrityLimits, Provenance, ServedCatalog,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// The registry document the extension was built with.
#[pyfunction]
pub(crate) fn bundled_model_cost_map(py: Python<'_>) -> Bound<'_, PyBytes> {
    PyBytes::new(py, BUNDLED_REGISTRY)
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct ModelCatalog {
    served: ServedCatalog,
}

impl From<ServedCatalog> for ModelCatalog {
    fn from(served: ServedCatalog) -> Self {
        Self { served }
    }
}

impl ModelCatalog {
    fn registry(&self) -> Option<&Catalog> {
        self.served.registry().map(Arc::as_ref)
    }
}

#[pymethods]
impl ModelCatalog {
    /// Parse a registry document. Raises `ValueError` when it is not a valid catalog.
    #[staticmethod]
    #[pyo3(signature = (body, source = None, revision = None, etag = None))]
    fn parse(
        py: Python<'_>,
        body: Vec<u8>,
        source: Option<String>,
        revision: Option<String>,
        etag: Option<String>,
    ) -> PyResult<Self> {
        let provenance = Provenance {
            source,
            revision,
            etag,
        };
        let registry = release_gil(py, || Catalog::parse(&body, provenance))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(ServedCatalog::new(Arc::new(registry)).into())
    }

    /// A catalog with no registry, serving only the entries registered on it.
    #[staticmethod]
    fn empty() -> Self {
        ServedCatalog::default().into()
    }

    /// Why the registry fails the integrity limits, or `None` when it passes them.
    fn integrity_failure(
        &self,
        reference_model_count: usize,
        min_model_count: usize,
        min_reference_ratio: f64,
    ) -> Option<String> {
        self.registry()?
            .check_size(
                reference_model_count,
                IntegrityLimits {
                    min_model_count,
                    min_reference_ratio,
                },
            )
            .err()
            .map(|error| error.to_string())
    }

    /// Raises `ValueError` naming every model and key with a bad value.
    fn register(
        &self,
        owner: &str,
        #[pyo3(from_py_with = from_py_argument)] entries: IndexMap<String, Fields>,
    ) -> PyResult<Self> {
        self.served
            .register(owner, entries)
            .map(Self::from)
            .map_err(|rejected| PyValueError::new_err(rejected.to_string()))
    }

    fn unregister(&self, owner: &str) -> Self {
        self.served.unregister(owner).into()
    }

    /// This catalog's registrations served on top of `registry`'s registry. Raises `ValueError`
    /// when a registration no longer reads cleanly on top of it.
    fn with_registry(&self, registry: &Self) -> PyResult<Self> {
        let replacement = registry
            .served
            .registry()
            .cloned()
            .ok_or_else(|| PyValueError::new_err("the replacement catalog has no registry"))?;
        self.served
            .with_registry(replacement)
            .map(Self::from)
            .map_err(|rejected| PyValueError::new_err(rejected.to_string()))
    }

    fn key_for(&self, name: &str) -> Option<String> {
        self.served.key_for(name).map(str::to_owned)
    }

    fn get(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        self.served
            .entry(key)
            .map(|fields| to_py(py, &fields))
            .transpose()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let entries: IndexMap<&str, Fields> = self
            .served
            .keys()
            .filter_map(|key| Some((key, self.served.entry(key)?)))
            .collect();
        to_py(py, &entries)
    }

    fn keys_list(&self) -> Vec<String> {
        self.served.keys().map(str::to_owned).collect()
    }

    /// The registry's fallback generalization rules, as the document wrote them.
    fn fallback_rules(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(
            py,
            self.registry()
                .and_then(Catalog::fallback_rules)
                .unwrap_or_default(),
        )
    }

    #[getter]
    fn registry_model_count(&self) -> usize {
        self.registry().map_or(0, Catalog::model_count)
    }

    #[getter]
    fn source(&self) -> Option<String> {
        self.registry()?.provenance().source.clone()
    }

    #[getter]
    fn revision(&self) -> Option<String> {
        self.registry()?.provenance().revision.clone()
    }

    #[getter]
    fn etag(&self) -> Option<String> {
        self.registry()?.provenance().etag.clone()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.served.contains(key)
    }

    fn __len__(&self) -> usize {
        self.served.keys().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::not_json(b"{".as_slice())]
    #[case::empty(b"{}".as_slice())]
    #[case::entry_not_object(br#"{"gpt-4o": 1}"#.as_slice())]
    #[case::bad_known_value(br#"{"gpt-4o": {"litellm_provider": "openai", "mode": "chatt"}}"#.as_slice())]
    fn parse_raises_value_error_for_an_invalid_registry(#[case] body: &[u8]) {
        Python::initialize();
        Python::attach(|py| {
            let error = ModelCatalog::parse(py, body.to_vec(), None, None, None)
                .err()
                .expect("invalid registry must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }
}
