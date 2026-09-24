//! The facade's read and write paths.

use litellm_cache::ExactCacheContext;
use litellm_cache_response::{CacheKeyInput, ResponseCacheRequest};
use litellm_host_python::{from_py, json_loads, release_gil, to_py};
use pyo3::{
    exceptions::PyException,
    prelude::*,
    sync::PyOnceLock,
    types::{PyBool, PyDict, PyFloat, PyString, PyTuple},
};
use serde_json::Value;

use super::{
    Binding, Cache, keys, override_of,
    steps::{Awaited, Continuation, Start},
    verbose_logger,
};
use crate::cache::{
    cache_error,
    request::{NativeRequest, duration, now},
};

fn pydantic_base_model(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static BASE_MODEL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    BASE_MODEL
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("pydantic")?.getattr("BaseModel")?.unbind())
        })
        .map(|class| class.bind(py))
}

pub(super) fn is_base_model(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(pydantic_base_model(value.py())?)
}

pub(super) fn uses_cache(slf: &Bound<'_, Cache>, kwargs: &Bound<'_, PyDict>) -> PyResult<bool> {
    match override_of(slf, "should_use_cache")? {
        Some(method) => Ok(method
            .call((), Some(kwargs))?
            .is(PyBool::new(slf.py(), true))),
        None => should_use_cache(slf, kwargs),
    }
}

pub(super) fn should_use_cache(
    slf: &Bound<'_, Cache>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let py = slf.py();
    let default_on = py
        .import("litellm.caching.caching")?
        .getattr("CacheMode")?
        .getattr("default_on")?;
    if slf.getattr("mode")?.eq(default_on)? {
        return Ok(true);
    }
    let control = kwargs
        .get_item("cache")?
        .unwrap_or_else(|| py.None().into_bound(py));
    keys::debug(
        py,
        "should_use_cache: kwargs: %s; _cache: %s",
        [kwargs.clone().into_any(), control.clone()],
    )?;
    if control.is_truthy()? && control.is_instance_of::<PyDict>() {
        let use_cache = control.call_method1("get", ("use-cache", false))?;
        return Ok(use_cache.is(PyBool::new(py, true)));
    }
    Ok(false)
}

pub(super) fn cache_key_for<'py>(
    slf: &Bound<'py, Cache>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    match kwargs.get_item("cache_key")? {
        Some(key) => Ok(key),
        None => match override_of(slf, "get_cache_key")? {
            Some(method) => method.call((), Some(kwargs)),
            None => keys::get_cache_key(slf, kwargs),
        },
    }
}

pub(super) fn cache_logic<'py>(
    slf: &Bound<'py, Cache>,
    cached_result: &Bound<'py, PyAny>,
    max_age: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(method) = override_of(slf, "_get_cache_logic")? else {
        return get_cache_logic(slf.py(), cached_result, max_age);
    };
    let kwargs = PyDict::new(slf.py());
    kwargs.set_item("cached_result", cached_result)?;
    kwargs.set_item("max_age", max_age)?;
    method.call((), Some(&kwargs))
}

pub(super) fn cache_entry<'py>(
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let Some(method) = override_of(slf, "_add_cache_logic")? else {
        return add_cache_logic(slf, result, kwargs);
    };
    let call_kwargs = kwargs.copy()?;
    call_kwargs.set_item("result", result)?;
    let (key, data, kwargs) = method.call((), Some(&call_kwargs))?.extract::<(
        Bound<'py, PyAny>,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
    )>()?;
    Ok((key, data, kwargs))
}

pub(super) fn get_cache_logic<'py>(
    py: Python<'py>,
    cached_result: &Bound<'py, PyAny>,
    max_age: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if cached_result.is_none()
        || !cached_result.is_instance_of::<PyDict>()
        || !cached_result.contains("timestamp")?
    {
        return Ok(cached_result.clone());
    }
    let timestamp = cached_result.get_item("timestamp")?;
    let current_time = py.import("time")?.call_method0("time")?;
    let response_age = current_time.sub(timestamp)?;
    if !max_age.is_none() && response_age.gt(max_age)? {
        return Ok(py.None().into_bound(py));
    }
    let cached_response = cached_result.call_method1("get", ("response",))?;
    if cached_response.is_instance_of::<PyDict>() {
        return Ok(cached_response);
    }
    match json_document(py, &cached_response) {
        Ok(decoded) => Ok(decoded),
        Err(_) => py
            .import("ast")?
            .call_method1("literal_eval", (cached_response,)),
    }
}

/// `json.loads(value)` for the `str`, `bytes` and `bytearray` values it accepts.
fn json_document<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let document = if let Ok(text) = value.cast::<PyString>() {
        text.to_str()?.as_bytes().to_vec()
    } else {
        value.extract::<Vec<u8>>()?
    };
    json_loads(py, &document).map(|decoded| decoded.into_bound(py))
}

pub(super) fn safe_lookup_kwargs<'py>(kwargs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let py = kwargs.py();
    let lookup = PyDict::new(py);
    for prompt_kwarg in ["messages", "input"] {
        if kwargs.contains(prompt_kwarg)? {
            lookup.set_item(prompt_kwarg, kwargs.get_item(prompt_kwarg)?)?;
        }
    }
    let metadata = kwargs.call_method1("get", ("metadata",))?;
    if let Ok(metadata) = metadata.cast::<PyDict>() {
        lookup.set_item("metadata", metadata.copy()?)?;
    }
    Ok(lookup)
}

pub(super) fn update_metadata_from_lookup(
    original_kwargs: &Bound<'_, PyAny>,
    cache_lookup_kwargs: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let original = original_kwargs.call_method1("get", ("metadata",))?;
    let lookup = cache_lookup_kwargs.call_method1("get", ("metadata",))?;
    let (Ok(original), Ok(lookup)) = (original.cast::<PyDict>(), lookup.cast::<PyDict>()) else {
        return Ok(());
    };
    if let Some(similarity) = lookup.get_item("semantic-similarity")? {
        original.set_item("semantic-similarity", similarity)?;
    }
    Ok(())
}

pub(super) fn stamp_semantic_similarity(
    kwargs: &Bound<'_, PyAny>,
    similarity: Option<f64>,
) -> PyResult<()> {
    let Some(similarity) = similarity else {
        return Ok(());
    };
    let metadata = kwargs.call_method1("get", ("metadata",))?;
    if let Ok(metadata) = metadata.cast::<PyDict>() {
        metadata.set_item("semantic-similarity", similarity)?;
    }
    Ok(())
}

fn duration_option(value: Option<Bound<'_, PyAny>>) -> Option<f64> {
    let value = value?;
    if value.is_instance_of::<PyBool>() {
        return None;
    }
    let seconds = value.extract::<f64>().ok()?;
    (seconds.is_finite() && seconds >= 0.0).then_some(seconds)
}

fn json_value(kwargs: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<Value>> {
    match kwargs.get_item(name)? {
        Some(value) if !value.is_none() => from_py(&value).map(Some),
        _ => Ok(None),
    }
}

pub(super) fn native_request(
    slf: &Bound<'_, Cache>,
    kwargs: &Bound<'_, PyDict>,
    key: &Bound<'_, PyAny>,
) -> PyResult<Option<NativeRequest>> {
    let py = slf.py();
    let Some(preset) = key.extract::<String>().ok().filter(|key| !key.is_empty()) else {
        return Ok(None);
    };
    let control = kwargs
        .get_item("cache")?
        .and_then(|control| control.cast_into::<PyDict>().ok())
        .unwrap_or_else(|| PyDict::new(py));
    let configured_ttl = slf.getattr("ttl")?;
    let configured_ttl = if configured_ttl.is_none() {
        duration_option(kwargs.get_item("ttl")?)
    } else {
        Some(configured_ttl.extract::<f64>()?)
    };
    let ttl = duration_option(control.get_item("ttl")?).or(configured_ttl);
    let max_age = duration_option(control.get_item("s-max-age")?)
        .or(duration_option(control.get_item("s-maxage")?));
    let scope = slf.getattr("semantic_cache_scope")?.extract::<String>()?;
    let key = CacheKeyInput {
        preset: Some(preset),
        ..Default::default()
    };
    let controls = ResponseCacheRequest::<ExactCacheContext>::new(key.clone()).controls;
    Ok(Some(NativeRequest {
        key,
        controls,
        ttl: ttl.map(duration).transpose()?,
        max_age: max_age.map(duration).transpose()?,
        messages: json_value(kwargs, "messages")?,
        input: json_value(kwargs, "input")?,
        metadata: json_value(kwargs, "metadata")?,
        litellm_metadata: json_value(kwargs, "litellm_metadata")?,
        litellm_params: json_value(kwargs, "litellm_params")?,
        scope: Some(scope),
    }))
}

pub(super) fn native_response(py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<Value> {
    if is_base_model(result)? {
        return from_py(&json_document(
            py,
            &result.call_method0("model_dump_json")?,
        )?);
    }
    if let Ok(text) = result.cast::<PyString>() {
        return match json_document(py, text.as_any()) {
            Ok(decoded) => from_py(&decoded),
            Err(_) => Ok(Value::String(text.to_str()?.to_owned())),
        };
    }
    from_py(result)
}

fn infinity(py: Python<'_>) -> Bound<'_, PyAny> {
    PyFloat::new(py, f64::INFINITY).into_any()
}

pub(super) fn get_cache(
    slf: &Bound<'_, Cache>,
    dynamic: Option<&Bound<'_, PyAny>>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    if !uses_cache(slf, kwargs)? {
        return Ok(py.None());
    }
    let key = cache_key_for(slf, kwargs)?;
    if key.is_none() {
        return Ok(py.None());
    }
    match slf.get().binding(py, slf.as_any(), dynamic)? {
        Binding::Native(service) => {
            let Some(request) = native_request(slf, kwargs, &key)? else {
                return Ok(py.None());
            };
            if keys::is_semantic_cache(slf)? {
                let lookup = release_gil(py, move || service.lookup_semantic(&request, now()))
                    .map_err(cache_error)?;
                stamp_semantic_similarity(kwargs.as_any(), lookup.similarity)?;
                return to_py(py, &lookup.value);
            }
            let response =
                release_gil(py, move || service.lookup(&request, now())).map_err(cache_error)?;
            to_py(py, &response)
        }
        Binding::Python(backend) => {
            let control = kwargs.call_method1("get", ("cache", PyDict::new(py)))?;
            let max_age = match (
                control.call_method1("get", ("s-maxage",))?,
                control.call_method1("get", ("s-max-age",))?,
            ) {
                (legacy, _) if legacy.is_truthy()? => legacy,
                (_, current) if current.is_truthy()? => current,
                _ => infinity(py),
            };
            let lookup_kwargs = safe_lookup_kwargs(kwargs.as_any())?;
            let cached =
                backend
                    .bind(py)
                    .call_method("get_cache", (&key,), Some(&lookup_kwargs))?;
            update_metadata_from_lookup(kwargs.as_any(), lookup_kwargs.as_any())?;
            cache_logic(slf, &cached, &max_age).map(Bound::unbind)
        }
    }
}

pub(super) fn async_get_cache<'py>(
    _awaited: Awaited,
    slf: &Bound<'py, Cache>,
    dynamic: Option<&Bound<'py, PyAny>>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Start<'py>> {
    let py = slf.py();
    if !uses_cache(slf, kwargs)? {
        return Ok(Start::none(py));
    }
    let key = cache_key_for(slf, kwargs)?;
    if key.is_none() {
        return Ok(Start::none(py));
    }
    match slf.get().binding(py, slf.as_any(), dynamic)? {
        Binding::Native(service) => {
            let Some(request) = native_request(slf, kwargs, &key)? else {
                return Ok(Start::none(py));
            };
            if keys::is_semantic_cache(slf)? {
                return Ok(Start::Await(
                    service.async_lookup_semantic_py(py, request)?,
                    Continuation::SemanticLookup {
                        kwargs: kwargs.clone().unbind(),
                    },
                ));
            }
            Ok(Start::Await(
                service.async_lookup_py(py, request)?,
                Continuation::NativeLookup,
            ))
        }
        Binding::Python(backend) => {
            let control = kwargs.call_method1("get", ("cache", PyDict::new(py)))?;
            let legacy = control.call_method1("get", ("s-maxage", infinity(py)))?;
            let max_age = control.call_method1("get", ("s-max-age", legacy))?;
            Ok(Start::Await(
                backend
                    .bind(py)
                    .call_method("async_get_cache", (&key,), Some(kwargs))?,
                Continuation::Lookup {
                    facade: slf.clone().into_any().unbind(),
                    max_age: max_age.unbind(),
                },
            ))
        }
    }
}

pub(super) fn add_cache_logic<'py>(
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let py = slf.py();
    let key = cache_key_for(slf, kwargs)?;
    if key.is_none() {
        return Err(PyException::new_err("cache key is None"));
    }
    let stored = if is_base_model(result)? {
        result.call_method0("model_dump_json")?
    } else {
        result.clone()
    };
    let ttl = slf.getattr("ttl")?;
    if !ttl.is_none() {
        kwargs.set_item("ttl", ttl)?;
    }
    if let Some(control) = kwargs.get_item("cache")?
        && let Ok(control) = control.cast::<PyDict>()
    {
        for (name, value) in control.iter() {
            if name.eq("ttl")? {
                kwargs.set_item("ttl", value)?;
            }
        }
    }
    let data = PyDict::new(py);
    data.set_item("timestamp", py.import("time")?.call_method0("time")?)?;
    data.set_item("response", stored)?;
    Ok((key, data, kwargs.clone()))
}

pub(super) fn add_cache(
    slf: &Bound<'_, Cache>,
    result: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let py = slf.py();
    if !uses_cache(slf, kwargs)? {
        return Ok(());
    }
    match slf.get().binding(py, slf.as_any(), None)? {
        Binding::Native(service) => {
            let key = cache_key_for(slf, kwargs)?;
            let Some(request) = native_request(slf, kwargs, &key)? else {
                return Ok(());
            };
            let response = native_response(py, result)?;
            release_gil(py, move || service.store(&request, response, now())).map_err(cache_error)
        }
        Binding::Python(backend) => {
            let (key, data, kwargs) = cache_entry(slf, result, kwargs)?;
            backend
                .bind(py)
                .call_method("set_cache", (key, data), Some(&kwargs))?;
            Ok(())
        }
    }
}

pub(super) fn batch_cache_write<'py>(
    _awaited: Awaited,
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let (key, data, kwargs) = cache_entry(slf, result, kwargs)?;
    slf.get().backend(py)?.into_bound(py).call_method(
        "batch_cache_write",
        (key, data),
        Some(&kwargs),
    )
}

pub(super) fn async_add_cache<'py>(
    awaited: Awaited,
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    dynamic: Option<&Bound<'py, PyAny>>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Start<'py>> {
    let py = slf.py();
    if !uses_cache(slf, kwargs)? {
        return Ok(Start::none(py));
    }
    let store = Continuation::Store {
        facade: slf.clone().into_any().unbind(),
    };
    match slf.get().binding(py, slf.as_any(), dynamic)? {
        Binding::Native(service) => {
            let key = cache_key_for(slf, kwargs)?;
            let Some(request) = native_request(slf, kwargs, &key)? else {
                return Ok(Start::none(py));
            };
            let response = native_response(py, result)?;
            Ok(Start::Await(
                service.async_store_py(py, request, response)?,
                store,
            ))
        }
        Binding::Python(target) => {
            let buffered = keys::type_name(slf)?.as_deref() == Some("redis")
                && !slf.getattr("redis_flush_size")?.is_none();
            if buffered {
                let write = match override_of(slf, "batch_cache_write")? {
                    Some(method) => {
                        let call_kwargs = kwargs.copy()?;
                        call_kwargs.set_item("result", result)?;
                        method.call((), Some(&call_kwargs))?
                    }
                    None => batch_cache_write(awaited, slf, result, kwargs)?,
                };
                return Ok(Start::Await(write, store));
            }
            let (key, data, kwargs) = cache_entry(slf, result, kwargs)?;
            Ok(Start::Await(
                target
                    .bind(py)
                    .call_method("async_set_cache", (key, data), Some(&kwargs))?,
                store,
            ))
        }
    }
}

pub(super) fn ping<'py>(_awaited: Awaited, slf: &Bound<'py, Cache>) -> PyResult<Start<'py>> {
    let py = slf.py();
    let ping = slf.get().backend(py)?.into_bound(py).getattr("ping")?;
    if !ping.is_truthy()? {
        return Ok(Start::none(py));
    }
    Ok(Start::Await(ping.call0()?, Continuation::Forward))
}

pub(super) fn delete_cache_keys<'py>(
    _awaited: Awaited,
    slf: &Bound<'py, Cache>,
    keys: &Bound<'py, PyAny>,
) -> PyResult<Start<'py>> {
    let py = slf.py();
    let delete = slf
        .get()
        .backend(py)?
        .into_bound(py)
        .getattr("delete_cache_keys")?;
    if !delete.is_truthy()? {
        return Ok(Start::none(py));
    }
    Ok(Start::Await(delete.call1((keys,))?, Continuation::Forward))
}

pub(super) fn disconnect<'py>(_awaited: Awaited, slf: &Bound<'py, Cache>) -> PyResult<Start<'py>> {
    let py = slf.py();
    let backend = slf.get().backend(py)?.into_bound(py);
    if !backend.hasattr("disconnect")? {
        return Ok(Start::none(py));
    }
    Ok(Start::Await(
        backend.call_method0("disconnect")?,
        Continuation::Discard,
    ))
}

pub(super) fn log_lookup_failure(py: Python<'_>, error: &PyErr) -> PyResult<()> {
    let formatted = py
        .import("traceback")?
        .call_method1("format_exception", (error.value(py),))?;
    let text = PyString::new(py, "")
        .call_method1("join", (formatted,))?
        .extract::<String>()?;
    py.import("litellm.caching.caching")?
        .getattr("print_verbose")?
        .call1((format!("An exception occurred: {text}"),))?;
    Ok(())
}

pub(super) fn log_add_cache_failure(slf: &Bound<'_, Cache>, error: &PyErr) -> PyResult<()> {
    let py = slf.py();
    let exception = error.value(py);
    if let Some(method) = override_of(slf, "_log_add_cache_failure")? {
        method.call1((exception,))?;
        return Ok(());
    }
    let message = "LiteLLM Cache: exception in add_cache";
    let backend = slf.get().backend(py)?;
    let caching_module = py.import("litellm.caching.caching")?;
    if backend
        .bind(py)
        .is_instance(&caching_module.getattr("RedisCache")?)?
    {
        let level = py.import("logging")?.getattr("ERROR")?;
        caching_module.getattr("log_redis_failure")?.call1((
            verbose_logger(py)?,
            level,
            message,
            exception,
        ))?;
        return Ok(());
    }
    verbose_logger(py)?.call_method1(
        "error",
        PyTuple::new(
            py,
            [
                PyString::new(py, "%s: %s").into_any(),
                PyString::new(py, message).into_any(),
                exception.clone().into_any(),
            ],
        )?,
    )?;
    Ok(())
}
