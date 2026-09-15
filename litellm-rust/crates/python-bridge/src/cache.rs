use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use litellm_cache::{
    BaseCache, CacheControls, CacheEntry, CacheKeyContext, CacheKeyField, CacheKeyInput, cache_key,
};
use litellm_cache_memory::InMemoryCache;
use litellm_core::call_lifecycle::cache::ResponseCachePlan;

#[pyclass]
struct NativeResponseCache {
    cache: Arc<InMemoryCache<CacheEntry>>,
}

#[derive(Default)]
struct CacheOptions {
    no_cache: bool,
    no_store: bool,
    use_cache: bool,
    ttl: Option<f64>,
    max_age: Option<f64>,
}

impl FromPyObject<'_, '_> for CacheOptions {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let values = obj.cast::<PyDict>()?.to_owned();
        Ok(Self {
            no_cache: is_true(&values, "no-cache")?,
            no_store: is_true(&values, "no-store")?,
            use_cache: is_true(&values, "use-cache")?,
            ttl: values
                .get_item("ttl")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract())
                .transpose()?,
            max_age: values
                .get_item("s-max-age")?
                .or(values.get_item("s-maxage")?)
                .filter(|value| !value.is_none())
                .map(|value| value.extract())
                .transpose()?,
        })
    }
}

pub(crate) fn snapshot(
    py: Python<'_>,
    call_type: &str,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<CacheControls> {
    let configured = py.import("litellm")?.getattr("cache")?;
    let standard = configured
        .get_type()
        .is(&py.import("litellm.caching.caching")?.getattr("Cache")?);
    let options = cache_options(kwargs.get_item("cache")?)?;
    let native_backend = standard
        && configured.getattr("cache")?.get_type().is(&py
            .import("litellm.caching.in_memory_cache")?
            .getattr("InMemoryCache")?);
    let supported_call_type = if standard {
        let supported = configured.getattr("supported_call_types")?;
        !supported.is_none()
            && supported
                .extract::<Vec<String>>()?
                .iter()
                .any(|value| value == call_type)
    } else {
        !matches!(call_type, "ocr" | "aocr")
    };
    Ok(CacheControls {
        configured: !configured.is_none(),
        supported_call_type,
        native_backend,
        default_on: !standard || configured.getattr("mode")?.extract::<String>()? == "default_on",
        caching: kwargs
            .get_item("caching")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract())
            .transpose()?,
        no_cache: options.no_cache,
        no_store: options.no_store,
        use_cache: options.use_cache,
    })
}

pub(crate) fn plan(
    py: Python<'_>,
    call_type: &str,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<ResponseCachePlan> {
    let controls = snapshot(py, call_type, kwargs)?;
    ResponseCachePlan {
        controls,
        ..Default::default()
    }
    .admit()
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if !controls.reads() && !controls.writes() {
        return Ok(ResponseCachePlan {
            controls,
            ..Default::default()
        });
    }
    let configured = py.import("litellm")?.getattr("cache")?;
    let backend = Some(native_cache(&configured)? as Arc<dyn BaseCache<Value = CacheEntry>>);
    let options = cache_options(kwargs.get_item("cache")?)?;
    let duration = |seconds: Option<f64>| -> PyResult<Option<Duration>> {
        seconds
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| PyValueError::new_err("cache duration must be finite and nonnegative"))
    };
    Ok(ResponseCachePlan {
        controls,
        backend,
        key: project_key(py, &configured, kwargs)?,
        ttl: duration(options.ttl.or(configured.getattr("ttl")?.extract()?))?,
        max_age: duration(options.max_age)?,
    })
}

fn native_cache(configured: &Bound<'_, PyAny>) -> PyResult<Arc<InMemoryCache<CacheEntry>>> {
    let attributes = configured.getattr("__dict__")?.cast_into::<PyDict>()?;
    let native = match attributes.get_item("_litellm_native_response_cache")? {
        Some(native) => native.cast_into::<NativeResponseCache>()?,
        None => {
            let native = Bound::new(
                configured.py(),
                NativeResponseCache {
                    cache: Arc::new(InMemoryCache::response_cache(
                        200,
                        Duration::from_secs(600),
                        1024 * 1024,
                    )),
                },
            )?;
            attributes.set_item("_litellm_native_response_cache", &native)?;
            native
        }
    };
    Ok(native.borrow().cache.clone())
}

fn dictionary<'py>(
    value: Option<Bound<'py, PyAny>>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    match value.filter(|value| !value.is_none()) {
        Some(value) => Ok(value.cast_into::<PyDict>()?),
        None => Ok(PyDict::new(py)),
    }
}

fn cache_options<'py>(value: Option<Bound<'py, PyAny>>) -> PyResult<CacheOptions> {
    value
        .filter(|value| !value.is_none())
        .map(|value| value.extract())
        .transpose()
        .map(|options| options.unwrap_or_default())
}

fn is_true(values: &Bound<'_, PyDict>, name: &str) -> PyResult<bool> {
    Ok(values
        .get_item(name)?
        .is_some_and(|value| value.is(&PyBool::new(values.py(), true))))
}

fn optional_text(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    value
        .filter(|value| !value.is_none())
        .map(|value| Ok(value.str()?.to_str()?.to_owned()))
        .transpose()
}

fn text_field(values: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    Ok(optional_text(values.get_item(name)?)?.filter(|value| !value.is_empty()))
}

fn project_key(
    py: Python<'_>,
    configured: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let parameters = dictionary(kwargs.get_item("litellm_params")?, py)?;
    if let Some(preset) = optional_text(parameters.get_item("preset_cache_key")?)? {
        return Ok(cache_key(&CacheKeyInput {
            preset: Some(preset),
            ..Default::default()
        }));
    }
    let api_params: std::collections::HashSet<String> = py
        .import("litellm.litellm_core_utils.model_param_helper")?
        .getattr("ModelParamHelper")?
        .call_method0("_get_all_llm_api_params")?
        .extract()?;
    let internal: Vec<String> = py
        .import("litellm.types.utils")?
        .getattr("all_litellm_params")?
        .extract()?;
    let include_provider_parameters: bool = py
        .import("litellm")?
        .getattr("enable_caching_on_provider_specific_optional_params")?
        .extract()?;
    let mut fields = Vec::new();
    for (name, value) in kwargs.iter() {
        let name: String = name.extract()?;
        let api_parameter = api_params.contains(&name);
        let internal_parameter = internal.contains(&name);
        if !api_parameter && (!include_provider_parameters || internal_parameter) {
            continue;
        }
        let value = if name == "file" {
            None
        } else {
            optional_text(Some(value))?
        };
        fields.push(CacheKeyField {
            name,
            value,
            api_parameter,
            internal_parameter,
        });
    }
    let metadata = dictionary(kwargs.get_item("metadata")?, py)?;
    let nested_metadata = dictionary(parameters.get_item("metadata")?, py)?;
    let dynamic = dictionary(kwargs.get_item("cache")?, py)?;
    let mut input = CacheKeyInput {
        fields,
        namespace: text_field(&dynamic, "namespace")?
            .or(text_field(&metadata, "redis_namespace")?)
            .or(optional_text(Some(configured.getattr("namespace")?))?),
        include_provider_parameters,
        ..Default::default()
    };
    let mut context = CacheKeyContext {
        model_group: text_field(&metadata, "model_group")?
            .or(text_field(&nested_metadata, "model_group")?),
        ..Default::default()
    };
    if let Some(groups) = metadata
        .get_item("caching_groups")?
        .filter(|value| !value.is_none())
    {
        for group in groups.try_iter()? {
            let group = group?;
            context
                .caching_groups
                .push((group.extract()?, group.str()?.to_str()?.to_owned()));
        }
    }
    if let Some(file) = kwargs.get_item("file")? {
        context.file_checksum = text_field(&metadata, "file_checksum")?;
        if context.file_checksum.is_none() {
            context.file_object_name = optional_text(
                file.getattr("name")
                    .or_else(|error| {
                        if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
                            Ok(py.None().into_bound(py))
                        } else {
                            Err(error)
                        }
                    })
                    .map(Some)?,
            )?;
        }
        context.metadata_file_name = text_field(&metadata, "file_name")?;
        context.parameters_file_name = text_field(&parameters, "file_name")?;
    }
    context.apply(&mut input);
    Ok(cache_key(&input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_options_extracts_controls_and_max_age_alias() {
        Python::initialize();
        Python::attach(|py| {
            let values = PyDict::new(py);
            values.set_item("no-cache", true).unwrap();
            values.set_item("use-cache", true).unwrap();
            values.set_item("ttl", 12.5).unwrap();
            values.set_item("s-maxage", 3.0).unwrap();

            let options: CacheOptions = values.extract().unwrap();

            assert!(options.no_cache);
            assert!(!options.no_store);
            assert!(options.use_cache);
            assert_eq!(options.ttl, Some(12.5));
            assert_eq!(options.max_age, Some(3.0));
        });
    }

    #[test]
    fn native_storage_is_reused_without_replacing_python_storage() {
        Python::initialize();
        Python::attach(|py| {
            let owner = py
                .import("types")
                .unwrap()
                .getattr("SimpleNamespace")
                .unwrap()
                .call0()
                .unwrap();
            let python_storage = PyDict::new(py);
            python_storage.set_item("python-only", "retained").unwrap();
            owner.setattr("cache", &python_storage).unwrap();
            let first = native_cache(&owner).unwrap();
            first
                .set_cache(
                    "native-only",
                    CacheEntry {
                        timestamp: 1.0,
                        response: serde_json::json!("native"),
                    },
                    None,
                )
                .unwrap();
            let second = native_cache(&owner).unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            assert!(second.get_cache("native-only").unwrap().is_some());
            assert!(owner.getattr("cache").unwrap().is(&python_storage));
            assert!(!python_storage.contains("native-only").unwrap());
            second.flush_cache().unwrap();
            assert_eq!(
                python_storage
                    .get_item("python-only")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "retained"
            );
            let weak = Arc::downgrade(&first);
            drop(first);
            drop(second);
            assert!(weak.upgrade().is_some());
            drop(owner);
            assert!(weak.upgrade().is_none());
        });
    }
}
