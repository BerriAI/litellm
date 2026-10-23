use std::convert::Infallible;

use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use litellm_http::transport::Error as TransportError;
use litellm_inference::Error;
use litellm_inference_tokenize::{Tokenize, TokenizeCall};
use litellm_llms_types::formats::tokenize::TokenizeResponse;
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::project::{MODULE, project_request};
use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::public_response,
    routes::RequestView,
};

pub(super) struct TokenizePythonHost {
    view: RequestView,
}

impl TokenizePythonHost {
    pub(super) fn new(view: RequestView) -> Self {
        Self { view }
    }

    fn map_failure(&self, py: Python<'_>, error: PyErr) -> PyResult<PyErr> {
        let mapped = py
            .import(MODULE)?
            .getattr("map_failure")?
            .call1((error.value(py), self.view.request(py)?))?;
        Ok(PyErr::from_value(mapped))
    }
}

impl PythonBinding for TokenizePythonHost {
    type Protocol = Tokenize;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<TokenizeCall, InvokeError<Error>> {
        project_request(py, &mut self.view, arguments).map_err(|error| {
            let mapped = self.map_failure(py, error).unwrap_or_else(|error| error);
            InvokeError::Python(mapped)
        })
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: TokenizeResponse,
    ) -> PyResult<Py<PyAny>> {
        public_response(py, MODULE, &response)
    }

    fn encode_stream_head(&mut self, _: Python<'_>, head: Infallible) -> PyResult<Py<PyAny>> {
        match head {}
    }

    fn encode_chunk(&mut self, _: Python<'_>, chunk: Infallible) -> PyResult<Py<PyAny>> {
        match chunk {}
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        if let Error::Secret(source) = &error
            && let Some(original) = crate::secrets::python_error(py, source.source_error())
        {
            return Ok(original);
        }
        let native = match error {
            Error::Transport(TransportError::Http { status, body }) => {
                let error = RustUpstreamError::new_err((status, body));
                error
                    .value(py)
                    .setattr("headers", Vec::<(String, String)>::new())?;
                error
            }
            other => route_error_to_pyerr(other),
        };
        self.map_failure(py, native)
    }

    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string().into())
    }
}

impl PythonHostCalls<Tokenize> for TokenizePythonHost {
    fn handle_host_call(
        &mut self,
        _: Python<'_>,
        op: Infallible,
    ) -> Result<(), InvokeError<Error>> {
        match op {}
    }
}

impl PythonOwned for TokenizePythonHost {
    fn close(&mut self, _: Python<'_>) {
        self.view.close();
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.view.traverse(visit)
    }
}
