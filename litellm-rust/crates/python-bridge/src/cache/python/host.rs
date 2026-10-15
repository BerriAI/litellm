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

struct Granted {
    access: Py<PyAny>,
    arguments: Py<PyDict>,
}

pub(crate) struct PythonCache {
    module: &'static str,
    granted: Option<Granted>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl PythonCache {
    pub(crate) fn new(asynchronous: bool) -> Self {
        Self {
            module: MODULE,
            granted: None,
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
            .import(self.module)?
            .getattr("access")?
            .call1((arguments, call_type))?;
        if access.is_none() {
            self.granted = None;
            return Ok(None);
        }
        let policy = CachePolicy {
            no_cache: !access.getattr("read")?.extract::<bool>()?,
            no_store: !access.getattr("write")?.extract::<bool>()?,
            ..CachePolicy::default()
        };
        self.granted = Some(Granted {
            access: access.unbind(),
            arguments: arguments.clone().unbind(),
        });
        Ok(Some(policy))
    }

    pub(crate) fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let granted = self.granted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("cache operation without granted cache access")
        })?;
        let access = granted.access.bind(py);
        let arguments = granted.arguments.bind(py);
        let function = |name: &str| py.import(self.module)?.getattr(name);
        let (pending, result) = match call {
            CacheCall::Lookup { reply } => (
                Pending::Lookup(reply),
                function(if self.asynchronous {
                    "alookup"
                } else {
                    "lookup"
                })
                .and_then(|lookup| lookup.call1((access, arguments))),
            ),
            CacheCall::Store { value, reply } => (
                Pending::Store(reply),
                function(if self.asynchronous { "astore" } else { "store" })
                    .and_then(|store| store.call1((access, to_py(py, &value)?, arguments))),
            ),
        };
        let result = result.map(Bound::unbind);
        self.pending = Some(pending);
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
        self.granted = None;
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(granted) = &self.granted else {
            return Ok(());
        };
        visit.call(&granted.access)?;
        visit.call(&granted.arguments)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use litellm_host::protocol::reply;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn a_cache_call_that_cannot_reach_python_answers_unavailable(
        #[values(false, true)] asynchronous: bool,
        #[values(false, true)] store: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let mut cache = PythonCache {
                module: "litellm_missing_cache_host",
                granted: Some(Granted {
                    access: py.None(),
                    arguments: PyDict::new(py).unbind(),
                }),
                pending: None,
                asynchronous,
            };
            let answer = if store {
                let (reply, answer) = reply();
                assert!(
                    cache
                        .begin(
                            py,
                            CacheCall::Store {
                                value: Value::Null,
                                reply
                            }
                        )
                        .unwrap()
                        .is_none()
                );
                answer.now_or_never().map(|answer| answer.unwrap().err())
            } else {
                let (reply, answer) = reply();
                assert!(
                    cache
                        .begin(py, CacheCall::Lookup { reply })
                        .unwrap()
                        .is_none()
                );
                answer.now_or_never().map(|answer| answer.unwrap().err())
            };
            assert!(matches!(answer, Some(Some(Error::Unavailable))));
        });
    }
}
