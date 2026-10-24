use crate::{Error, InvokeError, PythonOwned};
use litellm_host::{interceptors::ExecutionFacts, protocol::Protocol};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

/// A host lookup's progress: no answer, Python work to await first, or the call's answer.
pub enum Lookup<P: Protocol> {
    Miss,
    Await(Py<PyAny>),
    Hit(Hit<P>),
}

/// An answer the host found itself, delivered without starting the native machine.
pub struct Hit<P: Protocol> {
    pub facts: ExecutionFacts,
    pub output: HitOutput<P>,
}

pub enum HitOutput<P: Protocol> {
    Response(Py<PyAny>),
    Stream {
        head: P::StreamHead,
        chunks: Vec<P::Chunk>,
    },
}

/// Converts requests, responses, stream values and errors at the Python boundary.
pub trait PythonBinding: PythonOwned {
    type Protocol: Protocol<Error: std::fmt::Display>;

    /// The public exception a native failure maps to, kept as a value until the driver
    /// raises it.
    type Failure: Into<PyErr>;

    /// Decodes the keyword view returned by `prepare_arguments`, including preflight rewrites.
    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<
        <Self::Protocol as Protocol>::Request,
        InvokeError<<Self::Protocol as Protocol>::Error>,
    >;

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <Self::Protocol as Protocol>::Response,
    ) -> PyResult<Py<PyAny>>;

    /// A provider result as the public object Python hooks review, built from its JSON form.
    fn public_response(&mut self, py: Python<'_>, response: &Value) -> PyResult<Py<PyAny>> {
        crate::to_py(py, response)
    }

    /// Looks the decoded call up in the host's own store before the native machine starts.
    fn lookup(&mut self, _py: Python<'_>) -> PyResult<Lookup<Self::Protocol>> {
        Ok(Lookup::Miss)
    }

    /// Continues a lookup with the result of the Python work it awaited.
    fn resume_lookup(
        &mut self,
        _py: Python<'_>,
        _answer: PyResult<Py<PyAny>>,
    ) -> PyResult<Lookup<Self::Protocol>> {
        Err(Error::MissingState.into())
    }

    /// Keeps the response a provider call handed the caller, after its hooks reviewed it.
    fn store_response(&mut self, _py: Python<'_>, _response: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }

    /// Keeps the provider stream the caller read to its end.
    fn store_stream(&mut self, _py: Python<'_>) -> PyResult<()> {
        Ok(())
    }

    /// What the stream carries at hand-off, as the caller's stream receives it.
    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: <Self::Protocol as Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>>;

    /// One streamed chunk as the caller receives it.
    fn encode_chunk(
        &mut self,
        py: Python<'_>,
        chunk: <Self::Protocol as Protocol>::Chunk,
    ) -> PyResult<Py<PyAny>>;

    fn map_error(
        &self,
        py: Python<'_>,
        error: <Self::Protocol as Protocol>::Error,
    ) -> PyResult<Self::Failure>;

    fn host_error(error: &PyErr) -> <Self::Protocol as Protocol>::Error;
}
