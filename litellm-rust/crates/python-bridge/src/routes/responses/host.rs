use std::convert::Infallible;

use crate::routes::codec::{ProjectedCall, RouteCodec};
use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use litellm_inference_responses::{Error, route::Responses, types::ResponsesCall};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

pub(super) struct ResponsesPythonHost(pub RouteCodec);

impl From<ProjectedCall> for ResponsesCall {
    fn from(call: ProjectedCall) -> Self {
        Self {
            model: call.options.model,
            input: call.input,
            optional_params: call.params,
            connection: call.connection,
            api_key: call.options.api_key,
            api_base: call.options.api_base,
            custom_llm_provider: call.options.custom_llm_provider,
            extra_headers: call.options.extra_headers,
            timeout: call.options.timeout,
        }
    }
}

fn decode_previous_response_id(py: Python<'_>, call: ProjectedCall) -> PyResult<ProjectedCall> {
    let Some(serde_json::Value::String(encoded)) = call.params.get("previous_response_id") else {
        return Ok(call);
    };
    let decoded: String = py
        .import("litellm.responses.utils")?
        .getattr("ResponsesAPIRequestUtils")?
        .call_method1(
            "decode_previous_response_id_to_original_previous_response_id",
            (encoded,),
        )?
        .extract()?;
    let params = call
        .params
        .into_iter()
        .map(|(name, value)| match name.as_str() {
            "previous_response_id" => (name, serde_json::Value::String(decoded.clone())),
            _ => (name, value),
        })
        .collect();
    Ok(ProjectedCall { params, ..call })
}

impl PythonBinding for ResponsesPythonHost {
    type Protocol = Responses;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<ResponsesCall, InvokeError<Error>> {
        let call = self
            .0
            .project(py, arguments, "input")
            .and_then(|call| decode_previous_response_id(py, call))
            .map_err(InvokeError::Python)?;
        if call.streams() {
            return Err(InvokeError::Native(Error::Unsupported(
                "native Python responses streaming",
            )));
        }
        Ok(call.into())
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <Responses as litellm_host::protocol::Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        self.0.response(py, &response)
    }

    fn encode_stream_head(
        &mut self,
        _py: Python<'_>,
        head: <Responses as litellm_host::protocol::Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>> {
        let _ = head;
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native Python Responses streaming is not supported",
        ))
    }

    fn encode_chunk(
        &mut self,
        _py: Python<'_>,
        chunk: <Responses as litellm_host::protocol::Protocol>::Chunk,
    ) -> PyResult<Py<PyAny>> {
        let _ = chunk;
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native Python Responses streaming is not supported",
        ))
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        self.0.error(py, error)
    }
    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string().into())
    }
}

impl PythonHostCalls<Responses> for ResponsesPythonHost {
    fn handle_host_call(
        &mut self,
        _: Python<'_>,
        op: Infallible,
    ) -> Result<(), InvokeError<Error>> {
        match op {}
    }
}

impl PythonOwned for ResponsesPythonHost {
    fn close(&mut self, _: Python<'_>) {}
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0.bound)
    }
}
