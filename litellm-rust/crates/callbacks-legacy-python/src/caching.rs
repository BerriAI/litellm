use litellm_cache::{CacheControls, CacheKeyField, CacheKeyInput, cache_key};
use litellm_host_python::{from_py, to_py};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use crate::PublicCall;

pub(crate) struct Caching {
    backend: Py<PyAny>,
    pub key: String,
    pub controls: CacheControls,
    kwargs: Py<PyDict>,
    max_age: Option<f64>,
}

impl Caching {
    pub fn prepare(
        py: Python<'_>,
        call: &PublicCall,
        call_type: &str,
        asynchronous: bool,
    ) -> PyResult<Option<Self>> {
        let sdk = py.import("litellm")?;
        let cache = sdk.getattr("cache")?;
        if cache.is_none() {
            return Ok(None);
        }
        let supported = cache.getattr("supported_call_types")?;
        if supported.is_none() || !supported.contains(call_type)? {
            return Ok(None);
        }
        let kwargs = call.kwargs().bind(py).copy()?;
        let names = [
            "max_tokens",
            "messages",
            "model",
            "metadata",
            "stop_sequences",
            "stream",
            "system",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
            "container",
            "api_key",
            "api_base",
            "client",
            "custom_llm_provider",
        ];
        for (name, value) in names
            .iter()
            .filter(|name| !asynchronous || **name != "container")
            .zip(call.args().bind(py).iter())
        {
            kwargs.set_item(name, value)?;
        }
        let directives = kwargs
            .get_item("cache")?
            .unwrap_or_else(|| PyDict::new(py).into_any());
        let directive = |name: &str| directives.call_method1("get", (name,));
        let is_true =
            |value: Bound<'_, PyAny>| value.is(&true.into_pyobject(py).expect("bool conversion"));
        let controls = CacheControls {
            supported_call_type: true,
            configured: true,
            native_backend: false,
            default_on: cache.getattr("mode")?.eq("default_on")?,
            caching: kwargs
                .get_item("caching")?
                .filter(|v| !v.is_none())
                .map(is_true),
            no_cache: is_true(directive("no-cache")?),
            no_store: is_true(directive("no-store")?),
            use_cache: is_true(directive("use-cache")?),
        };
        if !controls.reads() && !controls.writes() {
            return Ok(None);
        }
        let metadata = kwargs
            .get_item("metadata")?
            .filter(|value| !value.is_none())
            .unwrap_or_else(|| PyDict::new(py).into_any());
        let params = kwargs
            .get_item("litellm_params")?
            .filter(|value| !value.is_none())
            .unwrap_or_else(|| PyDict::new(py).into_any());
        let preset = kwargs
            .get_item("cache_key")?
            .filter(|value| !value.is_none())
            .or_else(|| {
                params
                    .call_method1("get", ("preset_cache_key",))
                    .ok()
                    .filter(|value| !value.is_none())
            });
        let key = if let Some(preset) = preset {
            preset.extract()?
        } else {
            let api_params = py
                .import("litellm.litellm_core_utils.model_param_helper")?
                .getattr("ModelParamHelper")?
                .call_method0("_get_all_llm_api_params")?;
            let internal = py
                .import("litellm.types.utils")?
                .getattr("all_litellm_params")?;
            let include_provider_parameters = sdk
                .getattr("enable_caching_on_provider_specific_optional_params")?
                .is_truthy()?;
            let model_group = metadata.call_method1("get", ("model_group",))?;
            let nested_metadata = params.call_method1("get", ("metadata", PyDict::new(py)))?;
            let model_group = if model_group.is_truthy()? {
                model_group
            } else if nested_metadata.is_none() {
                py.None().into_bound(py)
            } else {
                nested_metadata.call_method1("get", ("model_group",))?
            };
            let groups = metadata.call_method1("get", ("caching_groups",))?;
            let mut group = None;
            if groups.is_truthy()? {
                for item in groups.try_iter()? {
                    let item = item?;
                    if item.contains(&model_group)? {
                        group = Some(item.str()?.to_string());
                        break;
                    }
                }
            }
            let fields = kwargs
                .iter()
                .map(|(name, value)| {
                    let name: String = name.extract()?;
                    let api_parameter = api_params.contains(&name)?;
                    let internal_parameter = internal.contains(&name)?;
                    let value = if value.is_none()
                        || (!api_parameter && (!include_provider_parameters || internal_parameter))
                    {
                        None
                    } else if name == "model" && group.is_some() {
                        group.clone()
                    } else if name == "model" && model_group.is_truthy()? {
                        Some(model_group.str()?.to_string())
                    } else {
                        Some(value.str()?.to_string())
                    };
                    Ok(CacheKeyField {
                        name,
                        value,
                        api_parameter,
                        internal_parameter,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let namespace = [
                directive("namespace")?,
                metadata.call_method1("get", ("redis_namespace",))?,
                cache.getattr("namespace")?,
            ]
            .into_iter()
            .find(|value| value.is_truthy().unwrap_or(false))
            .map(|value| value.extract())
            .transpose()?;
            cache_key(&CacheKeyInput {
                fields,
                preset: None,
                namespace,
                include_provider_parameters,
            })
        };
        let backend_kwargs = PyDict::new(py);
        for name in [
            "messages",
            "input",
            "metadata",
            "litellm_params",
            "parent_otel_span",
        ] {
            if let Some(value) = kwargs.get_item(name)? {
                backend_kwargs.set_item(name, value)?;
            }
        }
        let ttl = directive("ttl")?;
        let ttl = if !ttl.is_none() {
            Some(ttl)
        } else {
            let configured = cache.getattr("ttl")?;
            if configured.is_none() {
                kwargs.get_item("ttl")?
            } else {
                Some(configured)
            }
        };
        if let Some(ttl) = ttl {
            backend_kwargs.set_item("ttl", ttl)?;
        }
        let max_age = if asynchronous {
            directives.call_method1("get", ("s-max-age", directive("s-maxage")?))?
        } else {
            let first = directive("s-maxage")?;
            if first.is_truthy()? {
                first
            } else {
                directive("s-max-age")?
            }
        };
        Ok(Some(Self {
            backend: cache.getattr("cache")?.unbind(),
            key,
            controls,
            kwargs: backend_kwargs.unbind(),
            max_age: max_age.extract()?,
        }))
    }

    pub fn read(&self, py: Python<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
        self.backend
            .bind(py)
            .call_method(
                if asynchronous {
                    "async_get_cache"
                } else {
                    "get_cache"
                },
                (&self.key,),
                Some(self.kwargs.bind(py)),
            )
            .map(Bound::unbind)
    }

    pub fn decode(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
        if value.is_none() {
            return Ok(None);
        }
        let value: Value = from_py(value)?;
        let response = if let Some(timestamp) = value.get("timestamp").and_then(Value::as_f64) {
            if self
                .max_age
                .is_some_and(|max| litellm_host::event::epoch_seconds() - timestamp > max)
            {
                return Ok(None);
            }
            value.get("response").cloned().unwrap_or(Value::Null)
        } else {
            value
        };
        let response = match response {
            Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::Null),
            value => value,
        };
        Ok(response.is_object().then_some(response))
    }

    pub fn write(
        &self,
        py: Python<'_>,
        response: &Bound<'_, PyAny>,
        asynchronous: bool,
    ) -> PyResult<Py<PyAny>> {
        let entry = PyDict::new(py);
        entry.set_item("timestamp", litellm_host::event::epoch_seconds())?;
        entry.set_item("response", response)?;
        self.backend
            .bind(py)
            .call_method(
                if asynchronous {
                    "async_set_cache"
                } else {
                    "set_cache"
                },
                (&self.key, entry),
                Some(self.kwargs.bind(py)),
            )
            .map(Bound::unbind)
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.backend)?;
        visit.call(&self.kwargs)
    }
}

pub(crate) fn cached_public(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    to_py(py, value)
}
