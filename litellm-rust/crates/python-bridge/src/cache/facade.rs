use litellm_host_python::from_py;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyTuple, PyType},
};
use serde_json::Value;
use std::time::Duration;

use super::{NativeCacheHandle, native::NativeResponseCache};

struct ClassGuard {
    class: Py<PyType>,
    attributes: Vec<(String, Py<PyAny>)>,
}

struct ObjectGuard {
    reference: Py<PyAny>,
    classes: Vec<ClassGuard>,
    config_names: &'static [&'static str],
    config: Vec<Value>,
}

pub(super) struct FacadeGuard {
    outer: ObjectGuard,
    backend: ObjectGuard,
}

impl ObjectGuard {
    fn capture(
        py: Python<'_>,
        object: &Bound<'_, PyAny>,
        config_names: &'static [&'static str],
    ) -> PyResult<Self> {
        let classes = object
            .get_type()
            .getattr("__mro__")?
            .cast_into::<PyTuple>()?
            .iter()
            .map(|class| {
                let class = class.cast_into::<PyType>()?;
                let attributes = class
                    .getattr("__dict__")?
                    .call_method0("items")?
                    .try_iter()?
                    .map(|item| item?.extract::<(String, Py<PyAny>)>())
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(ClassGuard {
                    class: class.unbind(),
                    attributes,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let guard = Self {
            reference: py
                .import("weakref")?
                .getattr("ref")?
                .call1((object,))?
                .unbind(),
            classes,
            config_names,
            config: Self::config(object, config_names)?,
        };
        if !guard.matches(py, object)? {
            return Err(PyTypeError::new_err(
                "native facade registration requires unmodified built-in methods",
            ));
        }
        Ok(guard)
    }

    fn config(object: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<Vec<Value>> {
        names
            .iter()
            .map(|name| match object.getattr(*name) {
                Ok(value) => from_py(&value),
                Err(error)
                    if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(object.py()) =>
                {
                    Ok(Value::Null)
                }
                Err(error) => Err(error),
            })
            .collect()
    }

    fn matches(&self, py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.reference.bind(py).call0()?.is(object) {
            return Ok(false);
        }
        let mro = object
            .get_type()
            .getattr("__mro__")?
            .cast_into::<PyTuple>()?;
        if mro.len() != self.classes.len() {
            return Ok(false);
        }
        let instance = object.getattr("__dict__")?.cast_into::<PyDict>()?;
        for (class, expected) in mro.iter().zip(&self.classes) {
            if !class.is(expected.class.bind(py)) {
                return Ok(false);
            }
            let attributes = class.getattr("__dict__")?;
            if attributes.len()? != expected.attributes.len() {
                return Ok(false);
            }
            for (name, value) in &expected.attributes {
                if instance.contains(name)? || !attributes.get_item(name)?.is(value.bind(py)) {
                    return Ok(false);
                }
            }
        }
        Ok(Self::config(object, self.config_names)? == self.config)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)?;
        for class in &self.classes {
            visit.call(&class.class)?;
            for (_, value) in &class.attributes {
                visit.call(value)?;
            }
        }
        Ok(())
    }
}

impl FacadeGuard {
    pub(super) fn capture(
        py: Python<'_>,
        facade: &Bound<'_, PyAny>,
        kind: &str,
        native_default_ttl: Duration,
    ) -> PyResult<Self> {
        let cache_type = py.import("litellm.caching.caching")?.getattr("Cache")?;
        if !facade.get_type().is(&cache_type) {
            return Err(PyTypeError::new_err(
                "only exact built-in Cache facades can be registered",
            ));
        }
        let (module, name, cache_kind) = match kind {
            "memory" => ("litellm.caching.in_memory_cache", "InMemoryCache", "local"),
            "redis" => ("litellm.caching.redis_cache", "RedisCache", "redis"),
            _ => unreachable!(),
        };
        let backend = facade.getattr("cache")?;
        if facade.getattr("type")?.extract::<String>()? != cache_kind
            || !backend.get_type().is(&py.import(module)?.getattr(name)?)
        {
            return Err(PyTypeError::new_err(
                "facade and native backend types must match",
            ));
        }
        let python_default_ttl = backend.getattr("default_ttl")?.extract::<f64>()?;
        if python_default_ttl != native_default_ttl.as_secs_f64() {
            return Err(PyTypeError::new_err(
                "facade and native backend default TTLs must match",
            ));
        }
        Ok(Self {
            outer: ObjectGuard::capture(
                py,
                facade,
                &[
                    "type",
                    "mode",
                    "ttl",
                    "namespace",
                    "supported_call_types",
                    "redis_flush_size",
                ],
            )?,
            backend: ObjectGuard::capture(
                py,
                &backend,
                &[
                    "namespace",
                    "default_ttl",
                    "max_size_in_memory",
                    "max_size_per_item",
                ],
            )?,
        })
    }

    fn matches(&self, py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.outer.matches(py, facade)?
            && self.backend.matches(py, &facade.getattr("cache")?)?)
    }

    pub(super) fn traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.outer.traverse(&visit)?;
        self.backend.traverse(&visit)
    }
}

pub(super) fn resolve(
    py: Python<'_>,
    facade: &Bound<'_, PyAny>,
) -> PyResult<Option<NativeResponseCache>> {
    let Ok(dict) = facade
        .getattr("__dict__")
        .and_then(|dict| dict.cast_into::<PyDict>().map_err(Into::into))
    else {
        return Ok(None);
    };
    let Some(handle) = dict.get_item("_native_cache_handle")? else {
        return Ok(None);
    };
    let Ok(handle) = handle.extract::<PyRef<'_, NativeCacheHandle>>() else {
        return Ok(None);
    };
    let Some(guard) = &handle.guard else {
        return Ok(None);
    };
    if !guard.matches(py, facade).unwrap_or(false) {
        return Ok(None);
    }
    handle.service().map(Some)
}
