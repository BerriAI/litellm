use litellm_cache_redis::RedisTopology;
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
    max_connections: Option<usize>,
    attributes: RedisPoolAttributes,
}

struct RedisPoolAttributes {
    pool: &'static str,
    connection_class: &'static str,
    max_connections: Option<&'static str>,
}

const STANDALONE_POOL: RedisPoolAttributes = RedisPoolAttributes {
    pool: "connection_pool",
    connection_class: "connection_class",
    max_connections: Some("max_connections"),
};

const CLUSTER_POOL: RedisPoolAttributes = RedisPoolAttributes {
    pool: "nodes_manager",
    connection_class: "connection_pool_class",
    max_connections: None,
};

pub(super) struct FacadeGuard {
    outer: ObjectGuard,
    backend: ObjectGuard,
    redis_pool: Option<RedisPoolGuard>,
}

impl ObjectGuard {
    fn class_behaviors(class: &Bound<'_, PyType>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let py = class.py();
        let builtins = py.import("builtins")?;
        let property_type = builtins.getattr("property")?;
        let staticmethod_type = builtins.getattr("staticmethod")?;
        let classmethod_type = builtins.getattr("classmethod")?;
        class
            .getattr("__dict__")?
            .call_method0("items")?
            .try_iter()?
            .map(|item| {
                let item = item?;
                let (name, value): (String, Py<PyAny>) = item.extract()?;
                let value_bound = value.bind(py);
                let is_behavior = value_bound.is_callable()
                    || value_bound.is_instance(&property_type)?
                    || value_bound.is_instance(&staticmethod_type)?
                    || value_bound.is_instance(&classmethod_type)?;
                Ok(is_behavior.then_some((name, value)))
            })
            .filter_map(|result| match result {
                Ok(Some(attribute)) => Some(Ok(attribute)),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }

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
                let attributes = Self::class_behaviors(&class)?;
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
            let class = class.cast_into::<PyType>()?;
            if !class.is(expected.class.bind(py)) {
                return Ok(false);
            }
            let attributes = Self::class_behaviors(&class)?;
            if attributes.len() != expected.attributes.len() {
                return Ok(false);
            }
            for ((name, value), (expected_name, expected_value)) in
                attributes.iter().zip(&expected.attributes)
            {
                if name != expected_name
                    || instance.contains(name)?
                    || !value.bind(py).is(expected_value.bind(py))
                {
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
    fn capture(backend: &Bound<'_, PyAny>, attributes: RedisPoolAttributes) -> PyResult<Self> {
        let pool = backend.getattr("redis_client")?.getattr(attributes.pool)?;
        Ok(Self {
            reference: pool.clone().unbind(),
            connection_class: pool.getattr(attributes.connection_class)?.unbind(),
            connection_kwargs: pool
                .getattr("connection_kwargs")?
                .call_method0("copy")?
                .unbind(),
            max_connections: Self::max_connections(&pool, &attributes)?,
            attributes,
        })
    }

    fn max_connections(
        pool: &Bound<'_, PyAny>,
        attributes: &RedisPoolAttributes,
    ) -> PyResult<Option<usize>> {
        attributes
            .max_connections
            .map(|name| pool.getattr(name)?.extract::<usize>())
            .transpose()
    }

    fn matches(&self, py: Python<'_>, backend: &Bound<'_, PyAny>) -> PyResult<bool> {
        let pool = backend
            .getattr("redis_client")?
            .getattr(self.attributes.pool)?;
        Ok(self.reference.bind(py).is(&pool)
            && self
                .connection_class
                .bind(py)
                .is(&pool.getattr(self.attributes.connection_class)?)
            && self.max_connections == Self::max_connections(&pool, &self.attributes)?
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
        let cluster = matches!(service.topology(), Some(RedisTopology::Cluster { .. }));
        let (module, name, cache_kind) = match (kind, cluster) {
            ("memory", _) => ("litellm.caching.in_memory_cache", "InMemoryCache", "local"),
            ("redis", false) => ("litellm.caching.redis_cache", "RedisCache", "redis"),
            ("redis", true) => (
                "litellm.caching.redis_cluster_cache",
                "RedisClusterCache",
                "redis",
            ),
            ("qdrant_semantic", _) => (
                "litellm.caching.qdrant_semantic_cache",
                "QdrantSemanticCache",
                "qdrant-semantic",
            ),
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
        let backend_config_names = match kind {
            "memory" | "redis" => &[
                "namespace",
                "default_ttl",
                "max_size_in_memory",
                "max_size_per_item",
                "redis_kwargs",
                "redis_flush_size",
            ][..],
            "qdrant_semantic" => &[
                "qdrant_api_base",
                "qdrant_api_key",
                "collection_name",
                "similarity_threshold",
                "embedding_model",
                "vector_size",
                "embedding_max_input_tokens",
                "embedding_timeout",
            ][..],
            _ => unreachable!(),
        };
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
            backend: ObjectGuard::capture(py, &backend, backend_config_names)?,
            redis_pool: match (kind, cluster) {
                ("redis", false) => Some(RedisPoolGuard::capture(&backend, STANDALONE_POOL)?),
                ("redis", true) => Some(RedisPoolGuard::capture(&backend, CLUSTER_POOL)?),
                _ => None,
            },
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

#[cfg(test)]
mod tests {
    use super::ObjectGuard;
    use pyo3::{prelude::*, types::PyDict};

    #[test]
    fn class_data_shadowing_is_ignored_but_method_mutations_are_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"class Example:\n    data = 1\n    def method(self):\n        return 1\nobject = Example()\nobject.data = 2",
                None,
                Some(&namespace),
            )
            .unwrap();
            let object = namespace.get_item("object").unwrap().unwrap();
            let guard = ObjectGuard::capture(py, &object, &[]).unwrap();

            assert!(guard.matches(py, &object).unwrap());

            py.run(c"object.method = lambda: 2", None, Some(&namespace))
                .unwrap();
            assert!(!guard.matches(py, &object).unwrap());
        });
    }

    #[test]
    fn class_method_replacement_is_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"class Example:\n    def method(self):\n        return 1\nobject = Example()",
                None,
                Some(&namespace),
            )
            .unwrap();
            let object = namespace.get_item("object").unwrap().unwrap();
            let guard = ObjectGuard::capture(py, &object, &[]).unwrap();

            py.run(c"Example.method = lambda self: 2", None, Some(&namespace))
                .unwrap();
            assert!(!guard.matches(py, &object).unwrap());
        });
    }
}
