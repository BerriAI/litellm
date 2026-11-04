use std::sync::{LazyLock, PoisonError, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use litellm_model_catalog_refresh::{Origin, Provenance};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The installed cost map's provenance, as `get_model_cost_map_source_info` reports it
#[pyclass(
    frozen,
    module = "litellm.rust_bridge._native",
    name = "ModelCostMapProvenance"
)]
pub(crate) struct PyProvenance(Provenance);

#[pymethods]
impl PyProvenance {
    #[getter]
    fn source(&self) -> &'static str {
        match self.0.origin {
            Origin::Bundled { .. } => "local",
            Origin::Remote { .. } => "remote",
        }
    }

    #[getter]
    fn url(&self) -> Option<&str> {
        match &self.0.origin {
            Origin::Bundled { attempted } => attempted.as_deref(),
            Origin::Remote { url } => Some(url),
        }
    }

    #[getter]
    fn is_env_forced(&self) -> bool {
        self.0.is_env_forced
    }

    #[getter]
    fn fallback_reason(&self) -> Option<&str> {
        self.0.fallback_reason.as_deref()
    }

    /// Seconds since the Unix epoch
    #[getter]
    fn loaded_at(&self) -> f64 {
        self.0
            .loaded_at
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |since| since.as_secs_f64())
    }

    #[getter]
    fn revision(&self) -> Option<&str> {
        self.0.revision.as_deref()
    }

    #[getter]
    fn etag(&self) -> Option<&str> {
        self.0.etag.as_deref()
    }
}

static INSTALLED: LazyLock<RwLock<Option<Py<PyProvenance>>>> = LazyLock::new(RwLock::default);

/// Replaces the installed provenance, stamped now. Only the step that installs a map calls this
#[pyfunction]
#[pyo3(signature = (*, source, url, is_env_forced, fallback_reason, revision, etag))]
pub(crate) fn install_model_cost_map_provenance(
    py: Python<'_>,
    source: &str,
    url: Option<String>,
    is_env_forced: bool,
    fallback_reason: Option<String>,
    revision: Option<String>,
    etag: Option<String>,
) -> PyResult<()> {
    let origin = match (source, url) {
        ("local", attempted) => Origin::Bundled { attempted },
        ("remote", Some(url)) => Origin::Remote { url },
        _ => {
            return Err(PyValueError::new_err(format!(
                "a model cost map comes from 'local' or a 'remote' URL, got {source:?}"
            )));
        }
    };
    let provenance = Py::new(
        py,
        PyProvenance(Provenance {
            origin,
            is_env_forced,
            fallback_reason,
            loaded_at: SystemTime::now(),
            revision,
            etag,
        }),
    )?;
    *INSTALLED.write().unwrap_or_else(PoisonError::into_inner) = Some(provenance);
    Ok(())
}

/// `None` before the first install
#[pyfunction]
pub(crate) fn model_cost_map_provenance(py: Python<'_>) -> Option<Py<PyProvenance>> {
    INSTALLED
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .map(|provenance| provenance.clone_ref(py))
}
