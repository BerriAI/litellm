use litellm_cache::Error;
use litellm_host_python::{HostStep, InvokeError, from_py, to_py};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::service::CacheCall;

pub(crate) struct PythonCache {
    cache: Option<Py<PyAny>>,
    arguments: Option<Py<PyDict>>,
    asynchronous: bool,
}

impl PythonCache {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            cache: None,
            arguments: None,
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

    pub fn begin<E>(
        &mut self,
        py: Python<'_>,
        call: CacheCall,
    ) -> Result<HostStep<E>, InvokeError<E>> {
        let Some(cache) = self.cache.as_ref() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cache operation without configured cache",
            )
            .into());
        };
        let arguments = self
            .arguments
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("cache arguments unavailable")
            })?
            .bind(py)
            .copy()?;
        match call {
            CacheCall::Lookup { reply } => {
                let method = if self.asynchronous {
                    "async_get_cache"
                } else {
                    "get_cache"
                };
                let result = cache
                    .bind(py)
                    .call_method(method, (), Some(&arguments))
                    .map(Bound::unbind);
                HostStep::from_python(py, self.asynchronous, result, move |py, result| {
                    let answer = result.map_err(|_| Error::Unavailable).and_then(|value| {
                        if value.bind(py).is_none() {
                            return Ok(None);
                        }
                        from_py::<Value>(value.bind(py))
                            .map(Some)
                            .map_err(|_| Error::InvalidEntry)
                    });
                    reply.send(answer);
                    Ok(HostStep::Ready)
                })
            }
            CacheCall::Store { value, reply } => {
                let method = if self.asynchronous {
                    "async_add_cache"
                } else {
                    "add_cache"
                };
                let value = to_py(py, &value)?;
                let result = cache
                    .bind(py)
                    .call_method(method, (value,), Some(&arguments))
                    .map(Bound::unbind);
                HostStep::from_python(py, self.asynchronous, result, move |_, result| {
                    reply.send(result.map(|_| ()).map_err(|_| Error::Unavailable));
                    Ok(HostStep::Ready)
                })
            }
        }
    }

    pub fn close(&mut self) {
        self.cache = None;
        self.arguments = None;
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.cache)?;
        visit.call(&self.arguments)
    }
}
