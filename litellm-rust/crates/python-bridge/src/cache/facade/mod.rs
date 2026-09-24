//! `litellm.caching.caching.Cache` as a native class.

mod embedding;
mod entries;
mod init;
mod keys;
mod steps;

use std::sync::Arc;

use arc_swap::ArcSwapOption;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyAttributeError, PyException},
    prelude::*,
    sync::PyOnceLock,
    types::{PyDict, PyFrozenSet, PyString, PyTuple},
};

use self::steps::Operation;
use super::{guard::FacadeGuard, native::NativeResponseCache};

pub(super) struct NativeStorage {
    service: NativeResponseCache,
    guard: FacadeGuard,
    pid: u32,
}

impl NativeStorage {
    pub(super) fn new(service: NativeResponseCache, guard: FacadeGuard) -> Self {
        Self {
            service,
            guard,
            pid: std::process::id(),
        }
    }
}

pub(super) struct Storage {
    backend: Py<PyAny>,
    native: Option<NativeStorage>,
}

pub(super) enum Binding {
    Native(NativeResponseCache),
    Python(Py<PyAny>),
}

#[pyclass(
    frozen,
    subclass,
    dict,
    weakref,
    module = "litellm.caching.caching",
    name = "Cache"
)]
pub(crate) struct Cache {
    storage: ArcSwapOption<Storage>,
}

pub(super) const SEMANTIC_SCOPE_EXCLUDED_PARAMS: [&str; 3] = ["messages", "prompt", "input"];
pub(super) const SEMANTIC_TENANT_SCOPE_FIELDS: [&str; 3] = [
    "user_api_key",
    "user_api_key_team_id",
    "user_api_key_org_id",
];
pub(super) const SEMANTIC_END_USER_SCOPE_FIELD: &str = "user_api_key_end_user_id";

static ORIGINAL_METHODS: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

pub(crate) fn capture_method_table(py: Python<'_>) -> PyResult<()> {
    let table = py
        .get_type::<Cache>()
        .getattr("__dict__")?
        .call_method0("copy")?
        .cast_into::<PyDict>()?;
    ORIGINAL_METHODS.get_or_try_init(py, || Ok::<_, PyErr>(table.unbind()))?;
    Ok(())
}

pub(super) fn is_unmodified(py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<bool> {
    let class = py.get_type::<Cache>();
    if !facade.get_type().is(&class) {
        return Ok(false);
    }
    let Some(originals) = ORIGINAL_METHODS.get(py) else {
        return Ok(false);
    };
    let current = class.getattr("__dict__")?;
    let instance = facade.getattr("__dict__")?.cast_into::<PyDict>()?;
    for (name, original) in originals.bind(py).iter() {
        let name = name.cast_into::<PyString>()?;
        if name.to_str()?.starts_with("__") {
            continue;
        }
        if instance.contains(&name)? || !current.get_item(&name)?.is(&original) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn missing_backend() -> PyErr {
    PyAttributeError::new_err("'Cache' object has no attribute 'cache'")
}

pub(super) fn override_of<'py>(
    facade: &Bound<'py, Cache>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = facade.py();
    let class = py.get_type::<Cache>();
    let unmodified = facade.get_type().is(&class)
        && !facade
            .getattr("__dict__")?
            .cast_into::<PyDict>()?
            .contains(name)?
        && ORIGINAL_METHODS.get(py).is_some_and(|originals| {
            match (
                originals.bind(py).get_item(name).ok().flatten(),
                class
                    .getattr("__dict__")
                    .and_then(|table| table.get_item(name))
                    .ok(),
            ) {
                (Some(original), Some(current)) => current.is(&original),
                _ => false,
            }
        });
    if unmodified {
        return Ok(None);
    }
    facade.getattr(name).map(Some)
}

impl Cache {
    fn storage(&self) -> Option<Arc<Storage>> {
        self.storage.load_full()
    }

    pub(super) fn backend(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.storage()
            .map(|storage| storage.backend.clone_ref(py))
            .ok_or_else(missing_backend)
    }

    pub(super) fn set_backend(&self, backend: Py<PyAny>) -> Arc<Storage> {
        let published = Arc::new(Storage {
            backend,
            native: None,
        });
        self.storage.store(Some(Arc::clone(&published)));
        published
    }

    pub(super) fn bind_native(
        &self,
        py: Python<'_>,
        published: &Arc<Storage>,
        native: NativeStorage,
    ) {
        let bound = Arc::new(Storage {
            backend: published.backend.clone_ref(py),
            native: Some(native),
        });
        drop(self.storage.compare_and_swap(published, Some(bound)));
    }

    pub(super) fn native_service(
        &self,
        py: Python<'_>,
        facade: &Bound<'_, PyAny>,
    ) -> PyResult<Option<NativeResponseCache>> {
        let Some(storage) = self.storage() else {
            return Ok(None);
        };
        let Some(native) = &storage.native else {
            return Ok(None);
        };
        if native.pid != std::process::id() || !native.guard.matches(py, facade)? {
            return Ok(None);
        }
        Ok(Some(native.service.clone()))
    }

    pub(super) fn binding(
        &self,
        py: Python<'_>,
        facade: &Bound<'_, PyAny>,
        dynamic: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Binding> {
        if let Some(service) = self.native_service(py, facade)? {
            return Ok(Binding::Native(service));
        }
        if let Some(dynamic) = dynamic.filter(|object| !object.is_none()) {
            return Ok(Binding::Python(dynamic.clone().unbind()));
        }
        self.backend(py).map(Binding::Python)
    }
}

pub(super) fn kwargs_or_empty<'py>(
    py: Python<'py>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> Bound<'py, PyDict> {
    kwargs.cloned().unwrap_or_else(|| PyDict::new(py))
}

pub(super) fn verbose_logger(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(
                py.import("litellm._logging")?
                    .getattr("verbose_logger")?
                    .unbind(),
            )
        })
        .map(|logger| logger.bind(py))
}

fn triple<'py>(
    py: Python<'py>,
    key: Bound<'py, PyAny>,
    data: Bound<'py, PyDict>,
    kwargs: Bound<'py, PyDict>,
) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, [key, data.into_any(), kwargs.into_any()])?.unbind())
}

#[pymethods]
impl Cache {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn __new__(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            storage: ArcSwapOption::empty(),
        }
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        init::initialize(slf, args, kwargs)
    }

    #[getter(cache)]
    fn storage_object(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.backend(py)
    }

    #[setter(cache)]
    fn assign_cache(slf: &Bound<'_, Self>, backend: Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let published = slf.get().set_backend(backend.unbind());
        if !slf.hasattr("type")? {
            return Ok(());
        }
        match init::resolve_native(slf, &published) {
            Ok(()) => Ok(()),
            Err(error) if error.is_instance_of::<PyException>(py) => keys::debug(
                py,
                "LiteLLM Cache: the assigned storage object stays on Python: %s",
                [error.value(py).clone().into_any()],
            ),
            Err(error) => Err(error),
        }
    }

    #[classattr]
    #[pyo3(name = "_SEMANTIC_CACHE_SCOPE_EXCLUDED_PARAMS")]
    fn semantic_scope_excluded_params(py: Python<'_>) -> PyResult<Bound<'_, PyFrozenSet>> {
        PyFrozenSet::new(py, SEMANTIC_SCOPE_EXCLUDED_PARAMS)
    }

    #[classattr]
    #[pyo3(name = "_SEMANTIC_CACHE_TENANT_SCOPE_FIELDS")]
    fn semantic_tenant_scope_fields(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, SEMANTIC_TENANT_SCOPE_FIELDS)
    }

    #[classattr]
    #[pyo3(name = "_SEMANTIC_CACHE_END_USER_SCOPE_FIELD")]
    fn semantic_end_user_scope_field() -> &'static str {
        SEMANTIC_END_USER_SCOPE_FIELD
    }

    fn _is_semantic_cache(slf: &Bound<'_, Self>) -> PyResult<bool> {
        keys::is_semantic_cache(slf)
    }

    fn _semantic_cache_scope_fields<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(slf.py(), keys::semantic_scope_fields(slf)?)
    }

    fn _get_semantic_cache_tenant_scope(
        slf: &Bound<'_, Self>,
        kwargs: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        keys::semantic_tenant_scope(slf, kwargs)
    }

    #[pyo3(signature = (**kwargs))]
    fn get_cache_key(
        slf: &Bound<'_, Self>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        keys::get_cache_key(slf, &kwargs_or_empty(slf.py(), kwargs)).map(Bound::unbind)
    }

    fn _get_param_value(
        slf: &Bound<'_, Self>,
        param: &str,
        kwargs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        keys::param_value(slf, param, kwargs).map(Bound::unbind)
    }

    fn _get_model_param_value(
        slf: &Bound<'_, Self>,
        kwargs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        keys::model_param_value(slf, kwargs).map(Bound::unbind)
    }

    fn _get_caching_group(
        slf: &Bound<'_, Self>,
        metadata: &Bound<'_, PyAny>,
        model_group: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        keys::caching_group(slf.py(), metadata, model_group).map(Bound::unbind)
    }

    fn _get_file_param_value(
        slf: &Bound<'_, Self>,
        kwargs: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        keys::file_param_value(slf.py(), kwargs).map(Bound::unbind)
    }

    #[pyo3(signature = (**kwargs))]
    fn _get_preset_cache_key_from_kwargs(
        slf: &Bound<'_, Self>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        keys::preset_cache_key(&kwargs_or_empty(slf.py(), kwargs)).map(Bound::unbind)
    }

    #[pyo3(signature = (preset_cache_key, **kwargs))]
    fn _set_preset_cache_key_in_kwargs(
        slf: &Bound<'_, Self>,
        preset_cache_key: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        keys::set_preset_cache_key(preset_cache_key, &kwargs_or_empty(slf.py(), kwargs))
    }

    #[staticmethod]
    fn _get_hashed_cache_key(py: Python<'_>, cache_key: &str) -> PyResult<String> {
        keys::hashed_cache_key(py, cache_key)
    }

    #[pyo3(signature = (hash_hex, **kwargs))]
    fn _add_namespace_to_cache_key(
        slf: &Bound<'_, Self>,
        hash_hex: String,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        keys::add_namespace(slf, hash_hex, &kwargs_or_empty(slf.py(), kwargs))
    }

    fn generate_streaming_content(
        slf: &Bound<'_, Self>,
        content: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        slf.py()
            .import("litellm.caching.caching")?
            .getattr("_generate_streaming_content")?
            .call1((content,))
            .map(Bound::unbind)
    }

    fn _get_cache_logic(
        slf: &Bound<'_, Self>,
        cached_result: &Bound<'_, PyAny>,
        max_age: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        entries::get_cache_logic(slf.py(), cached_result, max_age).map(Bound::unbind)
    }

    #[staticmethod]
    fn _get_safe_cache_lookup_kwargs(kwargs: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
        entries::safe_lookup_kwargs(kwargs).map(Bound::unbind)
    }

    #[staticmethod]
    fn _update_metadata_from_cache_lookup_kwargs(
        original_kwargs: &Bound<'_, PyAny>,
        cache_lookup_kwargs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        entries::update_metadata_from_lookup(original_kwargs, cache_lookup_kwargs)
    }

    #[pyo3(signature = (dynamic_cache_object=None, **kwargs))]
    fn get_cache(
        slf: &Bound<'_, Self>,
        dynamic_cache_object: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let kwargs = kwargs_or_empty(py, kwargs);
        match entries::get_cache(slf, dynamic_cache_object, &kwargs) {
            Ok(value) => Ok(value),
            Err(error) => {
                entries::log_lookup_failure(py, &error)?;
                Ok(py.None())
            }
        }
    }

    #[pyo3(signature = (dynamic_cache_object=None, **kwargs))]
    fn async_get_cache<'py>(
        slf: &Bound<'py, Self>,
        dynamic_cache_object: Option<&Bound<'py, PyAny>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        steps::defer(
            py,
            Operation::GetCache {
                facade: slf.clone().unbind(),
                dynamic: dynamic_cache_object.map(|dynamic| dynamic.clone().unbind()),
                kwargs: kwargs_or_empty(py, kwargs).unbind(),
            },
        )
    }

    #[pyo3(signature = (result, **kwargs))]
    fn _add_cache_logic(
        slf: &Bound<'_, Self>,
        result: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let (key, data, kwargs) =
            entries::add_cache_logic(slf, result, &kwargs_or_empty(py, kwargs))?;
        triple(py, key, data, kwargs)
    }

    #[pyo3(signature = (result, **kwargs))]
    fn add_cache(
        slf: &Bound<'_, Self>,
        result: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let kwargs = kwargs_or_empty(py, kwargs);
        match entries::add_cache(slf, result, &kwargs) {
            Ok(()) => Ok(()),
            Err(error) => entries::log_add_cache_failure(slf, &error),
        }
    }

    fn _log_add_cache_failure(slf: &Bound<'_, Self>, exc: &Bound<'_, PyAny>) -> PyResult<()> {
        entries::log_add_cache_failure(slf, &PyErr::from_value(exc.clone()))
    }

    #[pyo3(signature = (result, dynamic_cache_object=None, **kwargs))]
    fn async_add_cache<'py>(
        slf: &Bound<'py, Self>,
        result: &Bound<'py, PyAny>,
        dynamic_cache_object: Option<&Bound<'py, PyAny>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        steps::defer(
            py,
            Operation::AddCache {
                facade: slf.clone().unbind(),
                result: result.clone().unbind(),
                dynamic: dynamic_cache_object.map(|dynamic| dynamic.clone().unbind()),
                kwargs: kwargs_or_empty(py, kwargs).unbind(),
            },
        )
    }

    #[pyo3(signature = (embedding_response, model, prompt_tokens=None, prompt_tokens_details=None))]
    fn _convert_to_cached_embedding(
        slf: &Bound<'_, Self>,
        embedding_response: &Bound<'_, PyAny>,
        model: &Bound<'_, PyAny>,
        prompt_tokens: Option<&Bound<'_, PyAny>>,
        prompt_tokens_details: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        embedding::cached_embedding(
            slf.py(),
            embedding_response,
            model,
            prompt_tokens,
            prompt_tokens_details,
        )
        .map(Bound::unbind)
    }

    fn _get_per_item_prompt_tokens_details(
        slf: &Bound<'_, Self>,
        result: &Bound<'_, PyAny>,
        idx_in_result_data: usize,
    ) -> PyResult<Py<PyAny>> {
        embedding::per_item_prompt_tokens_details(slf.py(), result, idx_in_result_data)
    }

    fn _get_per_item_prompt_tokens(
        slf: &Bound<'_, Self>,
        result: &Bound<'_, PyAny>,
        idx_in_result_data: usize,
    ) -> PyResult<Py<PyAny>> {
        embedding::per_item_prompt_tokens(slf.py(), result, idx_in_result_data)
    }

    #[pyo3(signature = (result, input, kwargs, idx_in_result_data=0))]
    fn add_embedding_response_to_cache(
        slf: &Bound<'_, Self>,
        result: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
        kwargs: &Bound<'_, PyDict>,
        idx_in_result_data: usize,
    ) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let (key, data, kwargs) =
            embedding::add_embedding_response(slf, result, input, kwargs, idx_in_result_data)?;
        triple(py, key, data, kwargs)
    }

    #[pyo3(signature = (result, dynamic_cache_object=None, **kwargs))]
    fn async_add_cache_pipeline<'py>(
        slf: &Bound<'py, Self>,
        result: &Bound<'py, PyAny>,
        dynamic_cache_object: Option<&Bound<'py, PyAny>>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        steps::defer(
            py,
            Operation::AddCachePipeline {
                facade: slf.clone().unbind(),
                result: result.clone().unbind(),
                dynamic: dynamic_cache_object.map(|dynamic| dynamic.clone().unbind()),
                kwargs: kwargs_or_empty(py, kwargs).unbind(),
            },
        )
    }

    #[pyo3(signature = (**kwargs))]
    fn should_use_cache(
        slf: &Bound<'_, Self>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        entries::should_use_cache(slf, &kwargs_or_empty(slf.py(), kwargs))
    }

    #[pyo3(signature = (result, **kwargs))]
    fn batch_cache_write<'py>(
        slf: &Bound<'py, Self>,
        result: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        steps::defer(
            py,
            Operation::BatchCacheWrite {
                facade: slf.clone().unbind(),
                result: result.clone().unbind(),
                kwargs: kwargs_or_empty(py, kwargs).unbind(),
            },
        )
    }

    fn ping<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        steps::defer(
            slf.py(),
            Operation::Ping {
                facade: slf.clone().unbind(),
            },
        )
    }

    fn delete_cache_keys<'py>(
        slf: &Bound<'py, Self>,
        keys: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        steps::defer(
            slf.py(),
            Operation::DeleteCacheKeys {
                facade: slf.clone().unbind(),
                keys: keys.clone().unbind(),
            },
        )
    }

    fn disconnect<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        steps::defer(
            slf.py(),
            Operation::Disconnect {
                facade: slf.clone().unbind(),
            },
        )
    }

    fn _supports_async(&self) -> bool {
        true
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let storage = self.storage.load();
        let Some(storage) = storage.as_ref() else {
            return Ok(());
        };
        visit.call(&storage.backend)?;
        if let Some(native) = &storage.native {
            native.service.traverse(&visit)?;
            native.guard.traverse(visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        self.storage.store(None);
    }
}
