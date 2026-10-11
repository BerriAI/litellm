use crate::{InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub struct StreamReplay<P: Protocol> {
    pub head: <P as Protocol>::StreamHead,
    pub chunks: Vec<<P as Protocol>::Chunk>,
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

    /// The complete response a finished stream described, for a route that can hold a
    /// stream back. The default refuses, so a hold-back policy on such a route fails the
    /// call before any chunk is swallowed.
    fn assemble_stream(
        &mut self,
        _py: Python<'_>,
        _chunks: Vec<<Self::Protocol as Protocol>::Chunk>,
    ) -> Result<
        <Self::Protocol as Protocol>::Response,
        InvokeError<<Self::Protocol as Protocol>::Error>,
    > {
        Err(InvokeError::Python(
            pyo3::exceptions::PyRuntimeError::new_err("this route cannot hold back a stream"),
        ))
    }

    /// The stream a caller would have seen for a complete public response: its head and
    /// every chunk, as the route's wire format replays one.
    fn stream_of_response(
        &mut self,
        _py: Python<'_>,
        _response: &Py<PyAny>,
    ) -> PyResult<StreamReplay<Self::Protocol>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "this route cannot synthesize a stream",
        ))
    }

    /// The chunk a held-back stream sends to keep the connection alive while it waits.
    fn keepalive_chunk(&mut self, _py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        Ok(None)
    }

    fn map_error(
        &self,
        py: Python<'_>,
        error: <Self::Protocol as Protocol>::Error,
    ) -> PyResult<Self::Failure>;

    fn host_error(error: &PyErr) -> <Self::Protocol as Protocol>::Error;
}
