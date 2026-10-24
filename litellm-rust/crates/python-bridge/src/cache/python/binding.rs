use std::convert::Infallible;

use litellm_host::{
    interceptors::{Cost, ExecutionFacts, ProviderIdentity, ResultSource},
    protocol::Protocol,
};
use litellm_host_python::{
    Hit, HitOutput, InvokeError, Lookup, PythonBinding, PythonHostCalls, PythonOwned,
    effective_py_args, from_py,
};
use litellm_inference::caching::successful_stream;
use pyo3::{
    exceptions::{PyException, PyRuntimeError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{IntoPyDict, PyDict},
};

const MODULE: &str = "litellm.rust_bridge.host.cache";

/// How a route's stream is kept in and replayed from the Python cache.
pub(crate) trait CachedStream: Protocol {
    /// The event a whole stream ends with, or `None` when the route never caches a stream.
    const TERMINAL_EVENT: Option<&'static str>;

    fn replay(text: String) -> Option<(Self::StreamHead, Vec<Self::Chunk>)>;
    fn bytes(chunk: &Self::Chunk) -> &[u8];
}

/// Wraps a route binding so the Python cache answers the call before the native machine
/// starts and keeps what the provider answered. Python owns the key, policy, entry format and
/// storage; this binding only carries values between the driver and
/// `litellm.rust_bridge.host.cache`.
pub(crate) struct CachedBinding<H> {
    inner: H,
    call_type: &'static str,
    asynchronous: bool,
    positional: Option<Py<PyDict>>,
    access: Option<Py<PyAny>>,
    captured: Vec<u8>,
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
            call_type,
            asynchronous,
            positional: Some(positional),
            access: None,
            captured: Vec::new(),
        }
    }

    fn call(&self, py: Python<'_>, name: &str, extra: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let Some(access) = &self.access else {
            return Ok(());
        };
        let function = py.import(MODULE)?.getattr(name)?;
        match extra {
            Some(value) => function.call1((access, value, self.asynchronous))?,
            None => function.call1((access, self.asynchronous))?,
        };
        Ok(())
    }
}

type ErrorOf<H> = <<H as PythonBinding>::Protocol as Protocol>::Error;

impl<H> CachedBinding<H>
where
    H: PythonBinding,
    H::Protocol: CachedStream,
{
    /// A lookup that failed in Python is a miss; a cancellation still ends the call.
    fn found(
        &mut self,
        py: Python<'_>,
        answer: PyResult<Py<PyAny>>,
    ) -> PyResult<Lookup<H::Protocol>> {
        let hit = match answer {
            Ok(hit) => hit,
            Err(error) if error.is_instance_of::<PyException>(py) => return Ok(Lookup::Miss),
            Err(error) => return Err(error),
        };
        let hit = hit.bind(py);
        if hit.is_none() {
            return Ok(Lookup::Miss);
        }
        let output = hit.getattr("output")?;
        let output = match output.getattr("kind")?.extract::<String>()?.as_str() {
            "Stream" => {
                match <H::Protocol as CachedStream>::replay(output.getattr("value")?.extract()?) {
                    Some((head, chunks)) => HitOutput::Stream { head, chunks },
                    None => return Ok(Lookup::Miss),
                }
            }
            _ => {
                let dumped = output.call_method(
                    "model_dump",
                    (),
                    Some(&[("mode", "json")].into_py_dict(py)?),
                )?;
                HitOutput::Response(
                    self.inner
                        .public_response(py, &from_py(&dumped.get_item("value")?)?)?,
                )
            }
        };
        Ok(Lookup::Hit(Hit {
            facts: ExecutionFacts {
                provider: ProviderIdentity {
                    model: hit.getattr("model")?.extract()?,
                    provider: hit.getattr("provider")?.extract()?,
                },
                source: ResultSource::Cache {
                    key: hit.getattr("key")?.extract()?,
                },
                cost: Cost::CacheHit,
            },
            output,
        }))
    }
}

impl<H> PythonBinding for CachedBinding<H>
where
    H: PythonBinding,
    H::Protocol: CachedStream,
{
    type Protocol = H::Protocol;
    type Failure = H::Failure;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<<Self::Protocol as Protocol>::Request, InvokeError<ErrorOf<H>>> {
        let request = self.inner.decode_request(py, arguments)?;
        let positional = self
            .positional
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("cache binding used after close"))?;
        let keyed = effective_py_args(positional.bind(py), arguments)?;
        let access = py
            .import(MODULE)?
            .getattr("access")?
            .call1((keyed, self.call_type))?;
        self.access = (!access.is_none()).then(|| access.unbind());
        Ok(request)
    }

    fn lookup(&mut self, py: Python<'_>) -> PyResult<Lookup<Self::Protocol>> {
        let Some(access) = &self.access else {
            return Ok(Lookup::Miss);
        };
        let lookup = py
            .import(MODULE)?
            .getattr(if self.asynchronous {
                "alookup"
            } else {
                "lookup"
            })?
            .call1((access,))
            .map(Bound::unbind);
        match lookup {
            Ok(awaitable) if self.asynchronous => Ok(Lookup::Await(awaitable)),
            answer => self.found(py, answer),
        }
    }

    fn resume_lookup(
        &mut self,
        py: Python<'_>,
        answer: PyResult<Py<PyAny>>,
    ) -> PyResult<Lookup<Self::Protocol>> {
        self.found(py, answer)
    }

    fn store_response(&mut self, py: Python<'_>, response: &Bound<'_, PyAny>) -> PyResult<()> {
        self.call(py, "store_response", Some(response.clone()))
    }

    fn store_stream(&mut self, py: Python<'_>) -> PyResult<()> {
        let captured = std::mem::take(&mut self.captured);
        let Some(terminal) = <H::Protocol as CachedStream>::TERMINAL_EVENT else {
            return Ok(());
        };
        let Ok(text) = String::from_utf8(captured) else {
            return Ok(());
        };
        if !successful_stream(&text, terminal) {
            return Ok(());
        }
        self.call(py, "store_stream", Some(text.into_pyobject(py)?.into_any()))
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
        if self.access.is_some() {
            self.captured
                .extend_from_slice(<H::Protocol as CachedStream>::bytes(&chunk));
        }
        self.inner.encode_chunk(py, chunk)
    }

    fn map_error(&self, py: Python<'_>, error: ErrorOf<H>) -> PyResult<H::Failure> {
        self.inner.map_error(py, error)
    }

    fn host_error(error: &PyErr) -> ErrorOf<H> {
        H::host_error(error)
    }
}

impl<H> PythonHostCalls<H::Protocol> for CachedBinding<H>
where
    H: PythonBinding,
    H::Protocol: CachedStream + Protocol<HostCall = Infallible>,
{
    fn handle_host_call(
        &mut self,
        _: Python<'_>,
        call: Infallible,
    ) -> Result<(), InvokeError<ErrorOf<H>>> {
        match call {}
    }
}

impl<H: PythonOwned> PythonOwned for CachedBinding<H> {
    fn close(&mut self, py: Python<'_>) {
        self.inner.close(py);
        self.positional = None;
        self.access = None;
        self.captured = Vec::new();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner.traverse(visit)?;
        if let Some(positional) = &self.positional {
            visit.call(positional)?;
        }
        if let Some(access) = &self.access {
            visit.call(access)?;
        }
        Ok(())
    }
}
