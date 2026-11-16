//! `get_llm_provider` over the Python tables `litellm.rust_bridge.host.provider_resolution` snapshots

use std::sync::{Arc, Mutex, PoisonError};

use litellm_core_utils::get_llm_provider_logic::{
    ProcessEnvironment, ProviderFailure, ProviderRequest, ProviderSnapshot, ProviderTables,
    Resolution, Resolved, get_llm_provider as resolve,
};
use litellm_host_python::from_py;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

const MODULE: &str = "litellm.rust_bridge.host.provider_resolution";

struct Cached {
    version: Py<PyAny>,
    tables: Arc<ProviderTables>,
}

static TABLES: Mutex<Option<Cached>> = Mutex::new(None);

fn host_function<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    cell.get_or_try_init(py, || {
        Ok::<_, PyErr>(py.import(MODULE)?.getattr(name)?.unbind())
    })
    .map(|function| function.bind(py).clone())
}

fn tables(py: Python<'_>) -> PyResult<Arc<ProviderTables>> {
    static VERSION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static SNAPSHOT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let version = host_function(py, &VERSION, "version")?.call0()?;
    let cached = TABLES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .map(|cached| (cached.version.clone_ref(py), Arc::clone(&cached.tables)));
    if let Some((cached_version, tables)) = cached
        && cached_version.bind(py).eq(&version)?
    {
        return Ok(tables);
    }
    let snapshot: ProviderSnapshot = from_py(&host_function(py, &SNAPSHOT, "snapshot")?.call0()?)?;
    let tables = Arc::new(ProviderTables::from(snapshot));
    *TABLES.lock().unwrap_or_else(PoisonError::into_inner) = Some(Cached {
        version: version.unbind(),
        tables: Arc::clone(&tables),
    });
    Ok(tables)
}

/// What `get_llm_provider(model, custom_llm_provider, api_base, api_key)` returns or raises, `Deferred` when only
/// Python can answer
pub(crate) fn resolution(py: Python<'_>, request: ProviderRequest<'_>) -> PyResult<Resolution> {
    let tables = tables(py)?;
    Ok(resolve(request, &tables, &ProcessEnvironment))
}

#[pyclass(
    frozen,
    get_all,
    name = "NativeLlmProvider",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeLlmProvider {
    model: String,
    custom_llm_provider: Option<String>,
    api_key: Option<String>,
    api_base: Option<String>,
}

#[pyclass(
    frozen,
    get_all,
    name = "NativeLlmProviderError",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeLlmProviderError {
    message: String,
    model: String,
    announce: bool,
}

fn text(value: Option<&Bound<'_, PyAny>>) -> Option<Option<String>> {
    match value {
        None => Some(None),
        Some(value) if value.is_none() => Some(None),
        Some(value) => value.extract::<String>().ok().map(Some),
    }
}

#[pyfunction]
#[pyo3(signature = (model, custom_llm_provider=None, api_base=None, api_key=None))]
pub(crate) fn get_llm_provider<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&Bound<'py, PyAny>>,
    api_base: Option<&Bound<'py, PyAny>>,
    api_key: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let (Some(Some(model)), Some(custom_llm_provider), Some(api_base), Some(api_key)) = (
        text(Some(model)),
        text(custom_llm_provider),
        text(api_base),
        text(api_key),
    ) else {
        return Ok(None);
    };
    let request = ProviderRequest {
        model: &model,
        custom_llm_provider: custom_llm_provider.as_deref(),
        api_base: api_base.as_deref(),
        api_key: api_key.as_deref(),
    };
    Ok(match resolution(py, request)? {
        Resolution::Resolved(Resolved {
            model,
            custom_llm_provider,
            api_key,
            api_base,
        }) => Some(
            Bound::new(
                py,
                NativeLlmProvider {
                    model,
                    custom_llm_provider,
                    api_key,
                    api_base,
                },
            )?
            .into_any(),
        ),
        Resolution::Failed(ProviderFailure {
            message,
            model,
            announce,
        }) => Some(
            Bound::new(
                py,
                NativeLlmProviderError {
                    message,
                    model,
                    announce,
                },
            )?
            .into_any(),
        ),
        Resolution::Deferred => None,
    })
}
