use litellm_cache::Error;
use litellm_host::protocol::Reply;
use litellm_host_python::{Awaiter, from_py, to_py};
use parking_lot::Mutex;
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use super::service::CacheCall;

enum Pending {
    Lookup(Reply<Result<Option<Value>, Error>>),
    Store(Reply<Result<(), Error>>),
}

struct Configured {
    cache: Py<PyAny>,
    arguments: Py<PyDict>,
}

pub(crate) struct PythonCache {
    configured: Mutex<Option<Configured>>,
    asynchronous: bool,
}

impl PythonCache {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            configured: Mutex::new(None),
            asynchronous,
        }
    }

    pub(in crate::cache) fn bind(&self, cache: Bound<'_, PyAny>, arguments: &Bound<'_, PyDict>) {
        *self.configured.lock() = Some(Configured {
            cache: cache.unbind(),
            arguments: arguments.clone().unbind(),
        });
    }

    /// Runs one cache operation against the configured Python cache, awaiting it on the
    /// caller's task for an async call. A cache failure is an answer to the route, a
    /// cancellation ends the call.
    pub async fn call(&self, task: &dyn Awaiter, call: CacheCall) -> PyResult<()> {
        let (pending, invoked) = Python::attach(|py| self.invoke(py, call))?;
        let result = match invoked {
            Ok(awaitable) if self.asynchronous => task.awaited(awaitable).await,
            invoked => invoked,
        };
        Python::attach(|py| answer(py, pending, result))
    }

    fn invoke(&self, py: Python<'_>, call: CacheCall) -> PyResult<(Pending, PyResult<Py<PyAny>>)> {
        let configured = self.configured.lock();
        let Some(configured) = configured.as_ref() else {
            return Err(PyRuntimeError::new_err(
                "cache operation without configured cache",
            ));
        };
        let arguments = configured.arguments.bind(py).copy()?;
        let cache = configured.cache.bind(py);
        let (pending, invoked) = match (call, self.asynchronous) {
            (CacheCall::Lookup { reply }, asynchronous) => (
                Pending::Lookup(reply),
                cache.call_method(
                    if asynchronous {
                        "async_get_cache"
                    } else {
                        "get_cache"
                    },
                    (),
                    Some(&arguments),
                ),
            ),
            (CacheCall::Store { value, reply }, asynchronous) => (
                Pending::Store(reply),
                cache.call_method(
                    if asynchronous {
                        "async_add_cache"
                    } else {
                        "add_cache"
                    },
                    (to_py(py, &value)?,),
                    Some(&arguments),
                ),
            ),
        };
        Ok((pending, invoked.map(Bound::unbind)))
    }

    pub fn close(&mut self) {
        *self.configured.get_mut() = None;
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(configured) = self.configured.try_lock() else {
            return Ok(());
        };
        let Some(configured) = configured.as_ref() else {
            return Ok(());
        };
        visit.call(&configured.cache)?;
        visit.call(&configured.arguments)
    }
}

fn answer(py: Python<'_>, pending: Pending, result: PyResult<Py<PyAny>>) -> PyResult<()> {
    if let Err(error) = &result
        && !error.is_instance_of::<PyException>(py)
    {
        return result.map(drop);
    }
    match pending {
        Pending::Lookup(reply) => {
            reply.send(result.map_err(|_| Error::Unavailable).and_then(|value| {
                if value.bind(py).is_none() {
                    Ok(None)
                } else {
                    from_py(value.bind(py))
                        .map(Some)
                        .map_err(|_| Error::InvalidEntry)
                }
            }))
        }
        Pending::Store(reply) => reply.send(result.map(drop).map_err(|_| Error::Unavailable)),
    }
    Ok(())
}
