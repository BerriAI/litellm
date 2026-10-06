use litellm_cache::Error;
use litellm_host::protocol::Reply;
use litellm_host_python::{from_py, to_py};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::service::CacheCall;

enum Pending {
    GetCacheKey(Reply<Result<String, Error>>),
    Lookup(Reply<Result<Option<Value>, Error>>),
    Store(Reply<Result<(), Error>>),
}

pub(crate) struct PythonCache {
    cache: Option<Py<PyAny>>,
    arguments: Option<Py<PyDict>>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl PythonCache {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            cache: None,
            arguments: None,
            pending: None,
            asynchronous,
        }
    }

    pub(in crate::cache) fn bind(
        &mut self,
        cache: Bound<'_, PyAny>,
        arguments: &Bound<'_, PyDict>,
    ) {
        self.cache = Some(cache.unbind());
        self.arguments = Some(arguments.clone().unbind());
    }

    pub fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let Some(cache) = self.cache.as_ref() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cache operation without configured cache",
            ));
        };
        let arguments = self
            .arguments
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("cache arguments unavailable")
            })?
            .bind(py)
            .copy()?;
        let await_result = self.asynchronous && !matches!(&call, CacheCall::GetCacheKey { .. });
        let (method, result) = match call {
            CacheCall::GetCacheKey { reply } => {
                if let Some(explicit) = arguments
                    .get_item("cache_key")?
                    .filter(|value| !value.is_none())
                {
                    self.pending = Some(Pending::GetCacheKey(reply));
                    return self.resume(py, Ok(explicit.unbind()));
                }
                self.pending = Some(Pending::GetCacheKey(reply));
                ("get_cache_key", None)
            }
            CacheCall::Lookup { key, reply } => {
                self.pending = Some(Pending::Lookup(reply));
                arguments.set_item("cache_key", key)?;
                (
                    if self.asynchronous {
                        "async_get_cache"
                    } else {
                        "get_cache"
                    },
                    None,
                )
            }
            CacheCall::Store { key, value, reply } => {
                self.pending = Some(Pending::Store(reply));
                arguments.set_item("cache_key", key)?;
                (
                    if self.asynchronous {
                        "async_add_cache"
                    } else {
                        "add_cache"
                    },
                    Some(to_py(py, &value)?),
                )
            }
        };
        let result = match result {
            Some(value) => cache
                .bind(py)
                .call_method(method, (value,), Some(&arguments)),
            None => cache.bind(py).call_method(method, (), Some(&arguments)),
        }
        .map(Bound::unbind);
        if await_result && result.is_ok() {
            return result.map(Some);
        }
        self.resume(py, result)
    }

    pub fn resume(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let Err(error) = &result
            && !error.is_instance_of::<pyo3::exceptions::PyException>(py)
        {
            self.pending = None;
            return Err(result.err().unwrap());
        }
        match self.pending.take() {
            Some(Pending::GetCacheKey(reply)) => {
                let key = result
                    .and_then(|value| value.bind(py).extract::<String>())
                    .map_err(|_| Error::Unavailable);
                reply.send(key);
            }
            Some(Pending::Lookup(reply)) => {
                let value = result.map_err(|_| Error::Unavailable).and_then(|value| {
                    if value.bind(py).is_none() {
                        Ok(None)
                    } else {
                        from_py(value.bind(py))
                            .map(Some)
                            .map_err(|_| Error::InvalidEntry)
                    }
                });
                reply.send(value);
            }
            Some(Pending::Store(reply)) => {
                reply.send(result.map(|_| ()).map_err(|_| Error::Unavailable));
            }
            None => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "cache reply without pending operation",
                ));
            }
        }
        Ok(None)
    }

    pub fn close(&mut self) {
        self.pending = None;
        self.cache = None;
        self.arguments = None;
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.cache)?;
        visit.call(&self.arguments)
    }
}
