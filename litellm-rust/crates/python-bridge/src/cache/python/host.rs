use litellm_cache::Error;
use litellm_host_python::{InvokeError, PythonTask, from_py, to_py};
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
}

impl PythonCache {
    pub fn new() -> Self {
        Self {
            cache: None,
            arguments: None,
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

    fn invoke(
        &self,
        py: Python<'_>,
        task: &PythonTask,
        sync_method: &str,
        async_method: &str,
        value: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
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
        let method = if task.asynchronous() {
            async_method
        } else {
            sync_method
        };
        match value {
            Some(value) => cache
                .bind(py)
                .call_method(method, (value,), Some(&arguments)),
            None => cache.bind(py).call_method(method, (), Some(&arguments)),
        }
        .map(Bound::unbind)
    }

    /// Answers one cache call through the selected Python cache, awaiting it on the
    /// caller's task. A cache failure is the call's answer, never the route's error.
    pub async fn call<E>(
        &mut self,
        task: &PythonTask,
        call: CacheCall,
    ) -> Result<(), InvokeError<E>> {
        match call {
            CacheCall::Lookup { reply } => {
                let result = Python::attach(|py| {
                    self.invoke(py, task, "get_cache", "async_get_cache", None)
                });
                let result = task.settle(result).await?;
                let answer = Python::attach(|py| {
                    result.map_err(|_| Error::Unavailable).and_then(|value| {
                        if value.bind(py).is_none() {
                            return Ok(None);
                        }
                        from_py::<Value>(value.bind(py))
                            .map(Some)
                            .map_err(|_| Error::InvalidEntry)
                    })
                });
                reply.send(answer);
                Ok(())
            }
            CacheCall::Store { value, reply } => {
                let result = Python::attach(|py| {
                    let value = to_py(py, &value)?;
                    self.invoke(py, task, "add_cache", "async_add_cache", Some(value))
                });
                let result = task.settle(result).await?;
                reply.send(result.map(|_| ()).map_err(|_| Error::Unavailable));
                Ok(())
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
