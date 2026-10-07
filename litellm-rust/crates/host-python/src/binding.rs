use litellm_host::{failure::Failure, protocol::Protocol};
use pyo3::{prelude::*, types::PyDict};

use crate::{InvokeError, PythonOwned};

/// What a host does with a native failure.
pub enum Settlement {
    /// The caller sees this exception, and the failure hooks run first.
    Fail(PyErr),
    /// The attempt is given up without failure hooks because another implementation will
    /// serve the call; the caller sees this exception.
    Abandon(PyErr),
}

/// Converts requests, responses, stream values and errors at the Python boundary.
pub trait PythonBinding: PythonOwned {
    type Protocol: Protocol<Error: std::fmt::Display>;

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

    /// What the host does with a native failure, decided from where the call failed and
    /// what the route reported.
    fn map_error(
        &self,
        py: Python<'_>,
        error: Failure<<Self::Protocol as Protocol>::Error>,
    ) -> PyResult<Settlement>;

    fn host_error(error: &PyErr) -> <Self::Protocol as Protocol>::Error;
}
