use super::service::{CacheCall, Cached};
use litellm_cache::Error;
use litellm_cache_response::{CacheHit, CachePolicy};
use litellm_host::protocol::{Protocol, Reply};
use litellm_host_python::{
    InvokeError, PythonBinding, PythonHostCalls, PythonOwned, effective_py_args, from_py, to_py,
};
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

const MODULE: &str = "litellm.rust_bridge.host.cache";

pub(crate) struct CachedBinding<H> {
    inner: H,
    cache: PythonCache,
    call_type: &'static str,
    positional: Option<Py<PyDict>>,
}

impl<H> CachedBinding<H> {
    pub(crate) fn new(
        inner: H,
        call_type: &'static str,
        asynchronous: bool,
        positional: Py<PyDict>,
    ) -> Self {
        Self {
            inner,
            cache: PythonCache::new(asynchronous),
            call_type,
            positional: Some(positional),
        }
    }
}

type ErrorOf<H> = <<H as PythonBinding>::Protocol as Protocol>::Error;

impl<H: PythonBinding> PythonBinding for CachedBinding<H> {
    type Protocol = Cached<H::Protocol>;
    type Failure = H::Failure;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<<Self::Protocol as Protocol>::Request, InvokeError<ErrorOf<H>>> {
        let request = self.inner.decode_request(py, arguments)?;
        let policy = self
            .positional
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("cache binding used after close"))
            .and_then(|positional| effective_py_args(positional.bind(py), arguments))
            .and_then(|keyed| self.cache.configure(py, &keyed, self.call_type))
            .map_err(InvokeError::Python)?;
        Ok((request, policy))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <Self::Protocol as Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        self.inner.encode_response(py, response)
    }

    fn public_response(
        &mut self,
        py: Python<'_>,
        response: &serde_json::Value,
    ) -> PyResult<Py<PyAny>> {
        self.inner.public_response(py, response)
    }

    fn response_value(
        &mut self,
        py: Python<'_>,
        public: &Bound<'_, PyAny>,
    ) -> PyResult<serde_json::Value> {
        self.inner.response_value(py, public)
    }

    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: <Self::Protocol as Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>> {
        self.inner.encode_stream_head(py, head)
    }

    fn encode_chunk(
        &mut self,
        py: Python<'_>,
        chunk: <Self::Protocol as Protocol>::Chunk,
    ) -> PyResult<Py<PyAny>> {
        self.inner.encode_chunk(py, chunk)
    }

    fn map_error(&self, py: Python<'_>, error: ErrorOf<H>) -> PyResult<H::Failure> {
        self.inner.map_error(py, error)
    }

    fn host_error(error: &PyErr) -> ErrorOf<H> {
        H::host_error(error)
    }
}

impl<H: PythonBinding> PythonHostCalls<Cached<H::Protocol>> for CachedBinding<H> {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        call: CacheCall,
    ) -> Result<(), InvokeError<ErrorOf<H>>> {
        self.begin_host_call(py, call).map(|_| ())
    }

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        call: CacheCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<ErrorOf<H>>> {
        self.cache.begin(py, call).map_err(InvokeError::Python)
    }

    fn resume_host_call(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<ErrorOf<H>>> {
        self.cache.resume(py, result).map_err(InvokeError::Python)
    }
}

impl<H: PythonOwned> PythonOwned for CachedBinding<H> {
    fn close(&mut self, py: Python<'_>) {
        self.inner.close(py);
        self.cache.close();
        self.positional = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner.traverse(visit)?;
        self.cache.traverse(visit)?;
        if let Some(positional) = &self.positional {
            visit.call(positional)?;
        }
        Ok(())
    }
}

enum Pending {
    Lookup(Reply<Result<Option<CacheHit>, Error>>),
    Store(Reply<Result<(), Error>>),
}

struct Granted {
    access: Py<PyAny>,
}

fn cache_hit(hit: &Bound<'_, PyAny>) -> PyResult<Option<CacheHit>> {
    if hit.is_none() {
        return Ok(None);
    }
    Ok(Some(CacheHit {
        value: from_py(&hit.getattr("response")?)?,
        key: hit.getattr("key")?.extract()?,
    }))
}

struct PythonCache {
    module: &'static str,
    granted: Option<Granted>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl PythonCache {
    fn new(asynchronous: bool) -> Self {
        Self {
            module: MODULE,
            granted: None,
            pending: None,
            asynchronous,
        }
    }

    fn configure(
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
        });
        Ok(Some(policy))
    }

    fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let granted = self.granted.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("cache operation without granted cache access")
        })?;
        let access = granted.access.bind(py);
        let function = |name: &str| py.import(self.module)?.getattr(name);
        let (pending, result) = match call {
            CacheCall::Lookup { reply } => (
                Pending::Lookup(reply),
                function(if self.asynchronous {
                    "alookup"
                } else {
                    "lookup"
                })
                .and_then(|lookup| lookup.call1((access,))),
            ),
            CacheCall::Store { value, reply } => (
                Pending::Store(reply),
                function(if self.asynchronous { "astore" } else { "store" })
                    .and_then(|store| store.call1((access, to_py(py, &value)?))),
            ),
        };
        let result = result.map(Bound::unbind);
        self.pending = Some(pending);
        if self.asynchronous && result.is_ok() {
            return result.map(Some);
        }
        self.resume(py, result)
    }

    fn resume(
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
                let hit = result
                    .map_err(|_| Error::Unavailable)
                    .and_then(|hit| cache_hit(hit.bind(py)).map_err(|_| Error::InvalidEntry));
                reply.send(hit);
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

    fn close(&mut self) {
        self.pending = None;
        self.granted = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(granted) = &self.granted else {
            return Ok(());
        };
        visit.call(&granted.access)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use litellm_host::protocol::reply;
    use rstest::rstest;
    use serde_json::Value;

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
                granted: Some(Granted { access: py.None() }),
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
