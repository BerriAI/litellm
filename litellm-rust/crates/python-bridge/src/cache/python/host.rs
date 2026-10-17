use litellm_cache::Error;
use litellm_cache_response::CachePolicy;
use litellm_host::protocol::Reply;
use litellm_host_python::{from_py, to_py};
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::service::CacheCall;

const MODULE: &str = "litellm.rust_bridge.host.cache";

enum Pending {
    Lookup(Reply<Result<Option<Value>, Error>>),
    Store(Reply<Result<(), Error>>),
}

struct Access {
    cache: Py<PyAny>,
    arguments: Py<PyDict>,
}

pub(crate) struct PythonCache {
    access: Option<Access>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl PythonCache {
    pub(crate) fn new(asynchronous: bool) -> Self {
        Self {
            access: None,
            pending: None,
            asynchronous,
        }
    }

    pub(crate) fn configure(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        call_type: &str,
    ) -> PyResult<Option<CachePolicy>> {
        let access = py
            .import(MODULE)?
            .getattr("access")?
            .call1((arguments, call_type))?;
        if access.is_none() {
            self.access = None;
            return Ok(None);
        }
        let policy = CachePolicy {
            no_cache: !access.getattr("read")?.extract::<bool>()?,
            no_store: !access.getattr("write")?.extract::<bool>()?,
            ..CachePolicy::default()
        };
        self.access = Some(Access {
            cache: access.getattr("cache")?.unbind(),
            arguments: arguments.clone().unbind(),
        });
        Ok(Some(policy))
    }

    pub(crate) fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let access = self.access.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("cache operation without granted cache access")
        })?;
        let module = py.import(MODULE)?;
        let cache = access.cache.bind(py);
        let arguments = access.arguments.bind(py);
        let (lookup, store) = if self.asynchronous {
            ("alookup", "astore")
        } else {
            ("lookup", "store")
        };
        let result = match call {
            CacheCall::Lookup { reply } => {
                self.pending = Some(Pending::Lookup(reply));
                module.getattr(lookup)?.call1((cache, arguments))
            }
            CacheCall::Store { value, reply } => {
                self.pending = Some(Pending::Store(reply));
                module
                    .getattr(store)?
                    .call1((cache, to_py(py, &value)?, arguments))
            }
        }
        .map(Bound::unbind);
        if self.asynchronous && result.is_ok() {
            return result.map(Some);
        }
        self.resume(py, result)
    }

    pub(crate) fn resume(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let Err(error) = &result
            && !error.is_instance_of::<PyException>(py)
        {
            self.pending = None;
            return Err(result.err().unwrap());
        }
        match self.pending.take() {
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
                return Err(PyRuntimeError::new_err(
                    "cache reply without pending operation",
                ));
            }
        }
        Ok(None)
    }

    pub(crate) fn close(&mut self) {
        self.pending = None;
        self.access = None;
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(access) = &self.access else {
            return Ok(());
        };
        visit.call(&access.cache)?;
        visit.call(&access.arguments)
    }
}
