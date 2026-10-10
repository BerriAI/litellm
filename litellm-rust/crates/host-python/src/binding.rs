use crate::{InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    fn map_error(
        &self,
        py: Python<'_>,
        error: <Self::Protocol as Protocol>::Error,
    ) -> PyResult<Self::Failure>;

    fn host_error(error: &PyErr) -> <Self::Protocol as Protocol>::Error;
}
