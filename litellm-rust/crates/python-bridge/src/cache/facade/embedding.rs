//! Embedding responses cached one input at a time.

use litellm_host_python::from_py;
use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
    types::{PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};
use serde_json::Value;

use super::{
    Binding, Cache,
    entries::{cache_entry, native_request, uses_cache},
    keys, override_of,
    steps::{Awaited, Continuation, Start},
};

pub(super) fn cached_embedding<'py>(
    py: Python<'py>,
    embedding_response: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
    prompt_tokens: Option<&Bound<'py, PyAny>>,
    prompt_tokens_details: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = if embedding_response.is_instance_of::<PyDict>() {
        embedding_response.clone()
    } else if embedding_response.hasattr("model_dump")? {
        embedding_response.call_method0("model_dump")?
    } else {
        py.import("builtins")?
            .getattr("vars")?
            .call1((embedding_response,))?
    };
    let field = |name: &str| data.call_method1("get", (name,));
    let cached = PyDict::new(py);
    let none = py.None().into_bound(py);
    let result: PyResult<()> = (|| {
        cached.set_item("embedding", field("embedding")?)?;
        cached.set_item("index", field("index")?)?;
        cached.set_item("object", field("object")?)?;
        cached.set_item("model", model)?;
        cached.set_item("prompt_tokens", prompt_tokens.unwrap_or(&none))?;
        cached.set_item(
            "prompt_tokens_details",
            prompt_tokens_details.unwrap_or(&none),
        )?;
        cached.set_item(
            "format_version",
            py.import("litellm.types.caching")?
                .getattr("EMBEDDING_CACHE_FORMAT_VERSION")?,
        )?;
        Ok(())
    })();
    match result {
        Ok(()) => Ok(cached),
        Err(error) if error.is_instance_of::<PyKeyError>(py) => Err(PyValueError::new_err(
            format!("Missing expected key in embedding response: {error}"),
        )),
        Err(error) => Err(error),
    }
}

fn usage_field<'py>(result: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let usage = result.getattr("usage")?;
    if usage.is_none() {
        return Ok(None);
    }
    let value = usage.getattr(name)?;
    Ok((!value.is_none()).then_some(value))
}

fn item_count(result: &Bound<'_, PyAny>) -> PyResult<usize> {
    result.getattr("data")?.len()
}

fn split(total: i64, count: i64, index: i64) -> i64 {
    let (quotient, remainder) = (total.div_euclid(count), total.rem_euclid(count));
    quotient + i64::from(index < remainder)
}

pub(super) fn per_item_prompt_tokens_details(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<Py<PyAny>> {
    let Some(details) = usage_field(result, "prompt_tokens_details")? else {
        return Ok(py.None());
    };
    let details_dict = if details.hasattr("model_dump")? {
        let kwargs = PyDict::new(py);
        kwargs.set_item("exclude_none", true)?;
        details
            .call_method("model_dump", (), Some(&kwargs))?
            .cast_into::<PyDict>()?
    } else if let Ok(details) = details.cast::<PyDict>() {
        let filtered = PyDict::new(py);
        for (key, value) in details.iter() {
            if !value.is_none() {
                filtered.set_item(key, value)?;
            }
        }
        filtered
    } else {
        return Ok(py.None());
    };
    if details_dict.is_empty() {
        return Ok(py.None());
    }
    let count = item_count(result)?;
    if count <= 1 {
        return Ok(details_dict.into_any().unbind());
    }
    let per_item = PyDict::new(py);
    for (key, value) in details_dict.iter() {
        if let Ok(whole) = value.cast::<PyInt>() {
            per_item.set_item(key, split(whole.extract()?, count as i64, index as i64))?;
        } else if let Ok(fraction) = value.cast::<PyFloat>() {
            per_item.set_item(key, fraction.extract::<f64>()? / count as f64)?;
        } else {
            per_item.set_item(key, value)?;
        }
    }
    Ok(per_item.into_any().unbind())
}

pub(super) fn per_item_prompt_tokens(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    index: usize,
) -> PyResult<Py<PyAny>> {
    let Some(total) = usage_field(result, "prompt_tokens")? else {
        return Ok(py.None());
    };
    let count = item_count(result)?;
    if count <= 1 {
        return Ok(total.unbind());
    }
    Ok(split(total.extract()?, count as i64, index as i64)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

pub(super) fn add_embedding_response<'py>(
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
    index: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let py = slf.py();
    let keyed = kwargs.copy()?;
    keyed.set_item("input", input)?;
    let preset_cache_key = match override_of(slf, "get_cache_key")? {
        Some(method) => method.call((), Some(&keyed))?,
        None => keys::get_cache_key(slf, &keyed)?,
    };
    kwargs.set_item("cache_key", &preset_cache_key)?;
    let embedding_response = result.getattr("data")?.get_item(index)?;
    let prompt_tokens = per_item_prompt_tokens(py, result, index)?;
    let prompt_tokens_details = per_item_prompt_tokens_details(py, result, index)?;
    let embedding_dict = cached_embedding(
        py,
        &embedding_response,
        &result.getattr("model")?,
        Some(prompt_tokens.bind(py)),
        Some(prompt_tokens_details.bind(py)),
    )?;
    cache_entry(slf, embedding_dict.as_any(), &kwargs.copy()?)
}

fn embedding_entry<'py>(
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
    index: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let Some(method) = override_of(slf, "add_embedding_response_to_cache")? else {
        return add_embedding_response(slf, result, input, kwargs, index);
    };
    let (key, data, kwargs) = method.call1((result, input, kwargs, index))?.extract::<(
        Bound<'py, PyAny>,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
    )>()?;
    Ok((key, data, kwargs))
}

pub(super) fn async_add_cache_pipeline<'py>(
    _awaited: Awaited,
    slf: &Bound<'py, Cache>,
    result: &Bound<'py, PyAny>,
    dynamic: Option<&Bound<'py, PyAny>>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Start<'py>> {
    let py = slf.py();
    if !uses_cache(slf, kwargs)? {
        return Ok(Start::none(py));
    }
    let input = kwargs
        .get_item("input")?
        .ok_or_else(|| PyKeyError::new_err("input"))?;
    let input_count = if input.is_instance_of::<PyList>() {
        input.len()?
    } else {
        1
    };
    let embeddings = item_count(result)?;
    if embeddings != input_count {
        keys::debug(
            py,
            "LiteLLM Cache: skipping embedding cache write, %d inputs but %d embeddings in the response",
            [
                input_count.into_pyobject(py)?.into_any(),
                embeddings.into_pyobject(py)?.into_any(),
            ],
        )?;
        return Ok(Start::none(py));
    }
    let ttl = slf.getattr("ttl")?;
    if !ttl.is_none() {
        kwargs.set_item("ttl", ttl)?;
    }
    let mut current = kwargs.clone();
    let mut cache_list = Vec::with_capacity(input_count);
    if input.is_instance_of::<PyList>() {
        for (index, item) in input.try_iter()?.enumerate() {
            let (key, data, next) = embedding_entry(slf, result, &item?, &current, index)?;
            current = next;
            cache_list.push((key, data));
        }
    } else if input.is_instance_of::<PyString>() {
        let (key, data, next) = embedding_entry(slf, result, &input, &current, 0)?;
        current = next;
        cache_list.push((key, data));
    }
    let store = Continuation::Store {
        facade: slf.clone().into_any().unbind(),
    };
    match slf.get().binding(py, slf.as_any(), dynamic)? {
        Binding::Native(service) => {
            let mut entries = Vec::with_capacity(cache_list.len());
            for (key, data) in &cache_list {
                let keyed = current.copy()?;
                keyed.set_item("cache_key", key)?;
                if let Some(request) = native_request(slf, &keyed, key)? {
                    let response: Value = from_py(&data.as_any().get_item("response")?)?;
                    entries.push((request, response));
                }
            }
            Ok(Start::Await(
                service.async_store_batch_py(py, entries)?,
                store,
            ))
        }
        Binding::Python(target) => {
            let pairs = cache_list
                .into_iter()
                .map(|(key, data)| PyTuple::new(py, [key, data.into_any()]))
                .collect::<PyResult<Vec<_>>>()?;
            let call_kwargs = current.copy()?;
            call_kwargs.set_item("cache_list", PyList::new(py, pairs)?)?;
            Ok(Start::Await(
                target
                    .bind(py)
                    .call_method("async_set_cache_pipeline", (), Some(&call_kwargs))?,
                store,
            ))
        }
    }
}
