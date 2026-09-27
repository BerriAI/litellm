use std::convert::Infallible;

use super::super::inference::InferenceHost;
use litellm_core::chat_completions::{Error, route::ChatCompletions, types::ChatCompletionsCall};
use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

pub(super) struct ChatCompletionsPythonHost(pub InferenceHost);

pub(super) fn project(
    host: &InferenceHost,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<ChatCompletionsCall> {
    let call = host.project(py, arguments, "messages")?;
    Ok(ChatCompletionsCall {
        model: call.options.model,
        messages: call.input,
        optional_params: call.params,
        api_key: call.options.api_key,
        api_base: call.options.api_base,
        custom_llm_provider: call.options.custom_llm_provider,
        extra_headers: call.options.extra_headers,
        timeout: call.options.timeout,
    })
}

impl PythonBinding for ChatCompletionsPythonHost {
    type Protocol = ChatCompletions;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<ChatCompletionsCall, InvokeError<Error>> {
        let call = project(&self.0, py, arguments).map_err(InvokeError::Python)?;
        if call
            .optional_params
            .get("stream")
            .is_some_and(|value| value == &serde_json::Value::Bool(true))
        {
            return Err(InvokeError::Native(Error::Unsupported(
                "native Python chat_completions streaming",
            )));
        }
        Ok(call)
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <ChatCompletions as litellm_host::protocol::Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        self.0.response(py, &response)
    }

    fn encode_stream_head(
        &mut self,
        _py: Python<'_>,
        head: <ChatCompletions as litellm_host::protocol::Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>> {
        match head {}
    }

    fn encode_chunk(
        &mut self,
        _py: Python<'_>,
        chunk: <ChatCompletions as litellm_host::protocol::Protocol>::Chunk,
    ) -> PyResult<Py<PyAny>> {
        match chunk {}
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        self.0.error(py, error)
    }
    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string())
    }
}

impl PythonHostCalls<ChatCompletions> for ChatCompletionsPythonHost {
    fn handle_host_call(
        &mut self,
        _: Python<'_>,
        op: Infallible,
    ) -> Result<(), InvokeError<Error>> {
        match op {}
    }
}

impl PythonOwned for ChatCompletionsPythonHost {
    fn close(&mut self, _: Python<'_>) {}
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0.request)
    }
}
