use litellm_host_python::from_py;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyTuple, PyType},
};
use serde_json::Value;

use super::{
    config::{CacheConfigProjection, NativeCacheConfig},
    handle::CacheTestHandle,
    native::NativeResponseCache,
};

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

struct RedisPoolGuard {
    reference: Py<PyAny>,
    connection_class: Py<PyAny>,
    connection_kwargs: Py<PyAny>,
    max_connections: usize,
}

pub(super) struct FacadeGuard {
    outer: ObjectGuard,
    backend: ObjectGuard,
    redis_pool: Option<RedisPoolGuard>,
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

impl RedisPoolGuard {
    fn capture(backend: &Bound<'_, PyAny>) -> PyResult<Self> {
        let pool = backend
            .getattr("redis_client")?
            .getattr("connection_pool")?;
        Ok(Self {
            reference: pool.clone().unbind(),
            connection_class: pool.getattr("connection_class")?.unbind(),
            connection_kwargs: pool
                .getattr("connection_kwargs")?
                .call_method0("copy")?
                .unbind(),
            max_connections: pool.getattr("max_connections")?.extract::<usize>()?,
        })
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        let pool = backend
            .getattr("redis_client")?
            .getattr("connection_pool")?;
        Ok(self.reference.bind(py).is(&pool)
            && self
                .connection_class
                .bind(py)
                .is(&pool.getattr("connection_class")?)
            && self.max_connections == pool.getattr("max_connections")?.extract::<usize>()?
            && self
                .connection_kwargs
                .bind(py)
                .eq(pool.getattr("connection_kwargs")?)?)
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reference)?;
        visit.call(&self.connection_class)?;
        visit.call(&self.connection_kwargs)
    }
}

impl FacadeGuard {
    pub(super) fn capture(
        py: Python<'_>,
        facade: &Bound<'_, PyAny>,
        service: &NativeResponseCache,
    ) -> PyResult<Self> {
        let kind = service.kind();
        let cache_type = py.import("litellm.caching.caching")?.getattr("Cache")?;
        if !facade.get_type().is(&cache_type) {
            return Err(PyTypeError::new_err(
                "only exact built-in Cache facades can be registered",
            ));
        }
        let (module, name, cache_kind) = match kind {
            "memory" => ("litellm.caching.in_memory_cache", "InMemoryCache", "local"),
            "redis" => ("litellm.caching.redis_cache", "RedisCache", "redis"),
            "gcs" => ("litellm.caching.gcs_cache", "GCSCache", "gcs"),
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
        let config = match NativeCacheConfig::project(facade)? {
            CacheConfigProjection::Native(config) => *config,
            CacheConfigProjection::Unsupported(reason) => {
                return Err(PyTypeError::new_err(reason.message()));
            }
        };
        if let Some(message) = config.service_mismatch(service) {
            return Err(PyTypeError::new_err(message));
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
                    "semantic_cache_scope",
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
                    "redis_kwargs",
                    "redis_flush_size",
                    "bucket_name",
                    "key_prefix",
                    "path_service_account",
                ],
            )?,
            redis_pool: (kind == "redis")
                .then(|| RedisPoolGuard::capture(&backend))
                .transpose()?,
        })
    }

    fn matches(&self, py: Python<'_>, facade: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.outer.matches(py, facade)? {
            return Ok(false);
        }
        let backend = facade.getattr("cache")?;
        if !self.backend.matches(py, &backend)? {
            return Ok(false);
        }
        match &self.redis_pool {
            Some(guard) => guard.matches(py, &backend),
            None => Ok(true),
        }
    }

    pub(super) fn traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.outer.traverse(&visit)?;
        self.backend.traverse(&visit)?;
        if let Some(guard) = &self.redis_pool {
            guard.traverse(&visit)?;
        }
        Ok(())
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
    let Ok(handle) = handle.extract::<PyRef<'_, CacheTestHandle>>() else {
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
