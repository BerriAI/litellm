use std::marker::PhantomData;

use litellm_cache_response::CachePolicy;
use litellm_host::protocol::Protocol;
use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::python::{CacheCall, PythonCache};

pub(crate) struct Cached<P>(PhantomData<P>);

impl<P: Protocol> Protocol for Cached<P> {
    type Request = (P::Request, Option<CachePolicy>);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
}

pub(crate) struct CachedHost<H> {
    inner: H,
    cache: PythonCache,
    call_type: &'static str,
}

impl<H> CachedHost<H> {
    pub(crate) fn new(inner: H, call_type: &'static str, asynchronous: bool) -> Self {
        Self {
            inner,
            cache: PythonCache::new(asynchronous),
            call_type,
        }
    }
}

type ErrorOf<H> = <<H as PythonBinding>::Protocol as Protocol>::Error;

impl<H: PythonBinding> PythonBinding for CachedHost<H> {
    type Protocol = Cached<H::Protocol>;
    type Failure = H::Failure;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<<Self::Protocol as Protocol>::Request, InvokeError<ErrorOf<H>>> {
        let request = self.inner.decode_request(py, arguments)?;
        let policy = self
            .cache
            .configure(py, arguments, self.call_type)
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

impl<H: PythonBinding> PythonHostCalls<Cached<H::Protocol>> for CachedHost<H> {
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

impl<H: PythonOwned> PythonOwned for CachedHost<H> {
    fn close(&mut self, py: Python<'_>) {
        self.inner.close(py);
        self.cache.close();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.inner.traverse(visit)?;
        self.cache.traverse(visit)
    }
}
