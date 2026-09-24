//! `Cache.get_cache_key` and its helpers.

use pyo3::{
    prelude::*,
    sync::PyOnceLock,
    types::{PyBool, PyDict, PyList, PyString, PyTuple},
};
use sha2::{Digest, Sha256};

use super::{
    Cache, SEMANTIC_END_USER_SCOPE_FIELD, SEMANTIC_SCOPE_EXCLUDED_PARAMS,
    SEMANTIC_TENANT_SCOPE_FIELDS, verbose_logger,
};

const SEMANTIC_TYPES: [&str; 3] = ["redis-semantic", "qdrant-semantic", "valkey-semantic"];

pub(super) fn debug<'py>(
    py: Python<'py>,
    message: &str,
    arguments: impl IntoIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<()> {
    let arguments = std::iter::once(PyString::new(py, message).into_any())
        .chain(arguments)
        .collect::<Vec<_>>();
    verbose_logger(py)?.call_method1("debug", PyTuple::new(py, arguments)?)?;
    Ok(())
}

pub(super) fn format_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    static FORMAT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    FORMAT
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("builtins")?.getattr("format")?.unbind())
        })?
        .bind(py)
        .call1((value,))?
        .extract()
}

pub(super) fn get<'py>(mapping: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    mapping.call_method1("get", (key,))
}

pub(super) fn or_empty_dict<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(if value.is_truthy()? {
        value
    } else {
        PyDict::new(py).into_any()
    })
}

fn first_truthy<'py>(candidates: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let mut last = None;
    for candidate in candidates {
        if candidate.is_truthy()? {
            return Ok(candidate);
        }
        last = Some(candidate);
    }
    last.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no candidates"))
}

pub(super) fn type_name(slf: &Bound<'_, Cache>) -> PyResult<Option<String>> {
    Ok(slf.getattr("type")?.extract::<String>().ok())
}

pub(super) fn is_semantic_cache(slf: &Bound<'_, Cache>) -> PyResult<bool> {
    Ok(type_name(slf)?.is_some_and(|name| SEMANTIC_TYPES.contains(&name.as_str())))
}

pub(super) fn semantic_scope_fields(slf: &Bound<'_, Cache>) -> PyResult<Vec<&'static str>> {
    let scope = slf.getattr("semantic_cache_scope")?;
    let end_user = scope
        .extract::<String>()
        .is_ok_and(|scope| scope == "end_user");
    Ok(SEMANTIC_TENANT_SCOPE_FIELDS
        .into_iter()
        .chain(end_user.then_some(SEMANTIC_END_USER_SCOPE_FIELD))
        .collect())
}

pub(super) fn semantic_tenant_scope(
    slf: &Bound<'_, Cache>,
    kwargs: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let py = slf.py();
    let litellm_params = or_empty_dict(py, get(kwargs, "litellm_params")?)?;
    let mut metadata_sources = Vec::with_capacity(4);
    for source in [kwargs, &litellm_params] {
        for key in ["metadata", "litellm_metadata"] {
            metadata_sources.push(or_empty_dict(py, get(source, key)?)?);
        }
    }
    let mut scope = String::new();
    for field in semantic_scope_fields(slf)? {
        let mut value = None;
        for source in &metadata_sources {
            let candidate = get(source, field)?;
            if !candidate.is_none() {
                value = Some(candidate);
                break;
            }
        }
        if let Some(value) = value {
            scope.push_str(field);
            scope.push_str(": ");
            scope.push_str(&format_value(py, &value)?);
        }
    }
    Ok(scope)
}

pub(super) fn preset_cache_key<'py>(kwargs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    let py = kwargs.py();
    if kwargs.is_empty() {
        return Ok(py.None().into_bound(py));
    }
    match kwargs.get_item("litellm_params")? {
        Some(litellm_params) => litellm_params.call_method1("get", ("preset_cache_key", py.None())),
        None => Ok(py.None().into_bound(py)),
    }
}

pub(super) fn set_preset_cache_key(
    preset_cache_key: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if kwargs.is_empty() {
        return Ok(());
    }
    if let Some(litellm_params) = kwargs.get_item("litellm_params")? {
        litellm_params.set_item("preset_cache_key", preset_cache_key)?;
    }
    Ok(())
}

pub(super) fn hashed_cache_key(py: Python<'_>, cache_key: &str) -> PyResult<String> {
    let hash_hex = format!("{:x}", Sha256::digest(cache_key.as_bytes()));
    debug(
        py,
        "Hashed cache key (SHA-256): %s",
        [PyString::new(py, &hash_hex).into_any()],
    )?;
    Ok(hash_hex)
}

pub(super) fn add_namespace(
    slf: &Bound<'_, Cache>,
    hash_hex: String,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let py = slf.py();
    let dynamic_cache_control = kwargs.call_method1("get", ("cache", PyDict::new(py)))?;
    let metadata = or_empty_dict(py, kwargs.call_method1("get", ("metadata",))?)?;
    let namespace = first_truthy(vec![
        get(&dynamic_cache_control, "namespace")?,
        get(&metadata, "redis_namespace")?,
        slf.getattr("namespace")?,
    ])?;
    let key = if namespace.is_truthy()? {
        format!("{}:{hash_hex}", format_value(py, &namespace)?)
    } else {
        hash_hex
    };
    debug(
        py,
        "Final hashed key: %s",
        [PyString::new(py, &key).into_any()],
    )?;
    Ok(key)
}

pub(super) fn caching_group<'py>(
    py: Python<'py>,
    metadata: &Bound<'py, PyAny>,
    model_group: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let caching_groups = metadata.call_method1("get", ("caching_groups", PyList::empty(py)))?;
    if !caching_groups.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    for group in caching_groups.try_iter()? {
        let group = group?;
        if group.contains(model_group)? {
            return Ok(group.str()?.into_any());
        }
    }
    Ok(py.None().into_bound(py))
}

pub(super) fn model_param_value<'py>(
    slf: &Bound<'py, Cache>,
    kwargs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let metadata = or_empty_dict(
        py,
        kwargs.call_method1("get", ("metadata", PyDict::new(py)))?,
    )?;
    let litellm_params = or_empty_dict(
        py,
        kwargs.call_method1("get", ("litellm_params", PyDict::new(py)))?,
    )?;
    let metadata_in_litellm_params = or_empty_dict(
        py,
        litellm_params.call_method1("get", ("metadata", PyDict::new(py)))?,
    )?;
    let model_group = first_truthy(vec![
        get(&metadata, "model_group")?,
        get(&metadata_in_litellm_params, "model_group")?,
    ])?;
    let group = caching_group(py, &metadata, &model_group)?;
    first_truthy(vec![group, model_group, kwargs.get_item("model")?])
}

pub(super) fn file_param_value<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let file = get(kwargs, "file")?;
    let metadata = kwargs.call_method1("get", ("metadata", PyDict::new(py)))?;
    let litellm_params = kwargs.call_method1("get", ("litellm_params", PyDict::new(py)))?;
    let file_name = file
        .getattr_opt("name")?
        .unwrap_or_else(|| py.None().into_bound(py));
    first_truthy(vec![
        get(&metadata, "file_checksum")?,
        file_name,
        get(&metadata, "file_name")?,
        get(&litellm_params, "file_name")?,
    ])
}

pub(super) fn param_value<'py>(
    slf: &Bound<'py, Cache>,
    param: &str,
    kwargs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match param {
        "model" => model_param_value(slf, kwargs),
        "file" => file_param_value(slf.py(), kwargs),
        _ => kwargs.get_item(param),
    }
}

pub(super) fn get_cache_key<'py>(
    slf: &Bound<'py, Cache>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let preset = preset_cache_key(kwargs)?;
    if !preset.is_none() {
        debug(py, "\nReturning preset cache key: %s", [preset.clone()])?;
        return Ok(preset);
    }
    let api_parameters = py
        .import("litellm.litellm_core_utils.model_param_helper")?
        .getattr("ModelParamHelper")?
        .call_method0("_get_all_llm_api_params")?;
    let litellm_parameters = py
        .import("litellm.types.utils")?
        .getattr("all_litellm_params")?;
    let provider_parameters = py
        .import("litellm")?
        .getattr("enable_caching_on_provider_specific_optional_params")?
        .is(PyBool::new(py, true));
    let semantic = is_semantic_cache(slf)?;
    let mut material = String::new();
    for (param, value) in kwargs.iter() {
        let name = param.extract::<String>()?;
        if semantic && SEMANTIC_SCOPE_EXCLUDED_PARAMS.contains(&name.as_str()) {
            continue;
        }
        if api_parameters.contains(&param)? {
            let param_value = param_value(slf, &name, kwargs.as_any())?;
            if !param_value.is_none() {
                material.push_str(&name);
                material.push_str(": ");
                material.push_str(&format_value(py, &param_value)?);
            }
        } else if !litellm_parameters.contains(&param)? && provider_parameters {
            if value.is_none() {
                continue;
            }
            material.push_str(&name);
            material.push_str(": ");
            material.push_str(&format_value(py, &value)?);
        }
    }
    if semantic {
        material.push_str(&semantic_tenant_scope(slf, kwargs.as_any())?);
    }
    let hashed = add_namespace(slf, hashed_cache_key(py, &material)?, kwargs)?;
    debug(
        py,
        "\nCreated cache key: %s (source material length: %d)",
        [
            PyString::new(py, &hashed).into_any(),
            material.chars().count().into_pyobject(py)?.into_any(),
        ],
    )?;
    let kwargs_for_preset = kwargs.copy()?;
    kwargs_for_preset.del_item("preset_cache_key").ok();
    let hashed = PyString::new(py, &hashed).into_any();
    set_preset_cache_key(&hashed, &kwargs_for_preset)?;
    Ok(hashed)
}
