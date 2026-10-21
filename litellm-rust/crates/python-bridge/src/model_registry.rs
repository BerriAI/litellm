//! The Python face of `litellm_model_catalog::Catalog`. Python builds `litellm.model_cost` from
//! `to_dict` and owns that dict from then on.
use std::sync::{Arc, LazyLock};

use litellm_host_python::{release_gil, to_py};
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

static BUNDLED: LazyLock<Result<Arc<Catalog>, String>> = LazyLock::new(|| {
    Catalog::parse(BUNDLED_REGISTRY)
        .map(Arc::new)
        .map_err(|error| error.to_string())
});

/// A parsed registry document. Every entry is kept as written, and the entries this build can't
/// read are listed in `problems`.
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct ModelRegistry {
    catalog: Arc<Catalog>,
}

#[pymethods]
impl ModelRegistry {
    /// Raises `ValueError` when `body` is not a JSON object with at least one model.
    #[staticmethod]
    fn parse(py: Python<'_>, body: PyBackedBytes) -> PyResult<Self> {
        let catalog = release_gil(py, || Catalog::parse(&body))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            catalog: Arc::new(catalog),
        })
    }

    /// The registry this build embeds.
    #[staticmethod]
    fn bundled() -> PyResult<Self> {
        BUNDLED
            .clone()
            .map(|catalog| Self { catalog })
            .map_err(PyValueError::new_err)
    }

    /// The bytes of the registry this build embeds.
    #[staticmethod]
    fn bundled_bytes(py: Python<'_>) -> Bound<'_, PyBytes> {
        PyBytes::new(py, BUNDLED_REGISTRY)
    }

    /// Every problem, each naming the model and the key it was found at.
    fn problems(&self) -> Vec<String> {
        self.catalog
            .problems()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// A new dict of every top-level key and value as the document wrote them, in document order.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &Document(&self.catalog))
    }
}

struct Document<'a>(&'a Catalog);

impl serde::Serialize for Document<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.written())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    #[rstest]
    #[case::not_json(b"{".as_slice())]
    #[case::not_an_object(b"[]".as_slice())]
    #[case::empty(b"{}".as_slice())]
    #[case::only_sections(br#"{"sample_spec": {}}"#.as_slice())]
    fn parse_raises_value_error_for_a_document_with_no_models(#[case] body: &[u8]) {
        Python::initialize();
        Python::attach(|py| {
            let bytes = PyBytes::new(py, body).extract::<PyBackedBytes>().unwrap();
            let error = ModelRegistry::parse(py, bytes).err().unwrap();
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[rstest]
    fn to_dict_keeps_bad_entries_and_sections_in_document_order() {
        Python::initialize();
        Python::attach(|py| {
            let body =
                br#"{"b":{"litellm_provider":"t","mode":"chatt"},"sample_spec":{"x":1},"a":4}"#;
            let bytes = PyBytes::new(py, body).extract::<PyBackedBytes>().unwrap();
            let registry = ModelRegistry::parse(py, bytes).unwrap();
            let dict = registry.to_dict(py).unwrap().into_bound(py);
            let dict = dict.cast::<PyDict>().unwrap();
            let entry = dict.get_item("b").unwrap().unwrap();

            assert!(dict.is_exact_instance_of::<PyDict>());
            assert_eq!(
                dict.keys().extract::<Vec<String>>().unwrap(),
                ["b", "sample_spec", "a"]
            );
            assert_eq!(
                dict.get_item("a")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                4
            );
            assert_eq!(
                entry.get_item("mode").unwrap().extract::<String>().unwrap(),
                "chatt"
            );
            assert_eq!(registry.problems().len(), 2);
        });
    }
}
