use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use litellm_inference_chat::{Error, route::ChatCompletions, types::ChatCompletionsCall};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

use super::super::inference::InferenceHost;
use crate::cache::{CacheCall, PythonCache, PythonCacheConfig, PythonCached};

pub(super) struct ChatCompletionsPythonHost {
    host: InferenceHost,
    cache: PythonCache,
    call_type: &'static str,
}

impl ChatCompletionsPythonHost {
    pub(super) fn new(host: InferenceHost, asynchronous: bool, call_type: &'static str) -> Self {
        Self {
            host,
            cache: PythonCache::new(asynchronous),
            call_type,
        }
    }
}

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
    type Protocol = PythonCached<ChatCompletions>;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<(ChatCompletionsCall, Option<PythonCacheConfig>), InvokeError<Error>> {
        let call = project(&self.host, py, arguments).map_err(InvokeError::Python)?;
        if call
            .optional_params
            .get("stream")
            .is_some_and(|value| value == &serde_json::Value::Bool(true))
        {
            return Err(InvokeError::Native(Error::Unsupported(
                "native Python chat_completions streaming",
            )));
        }
        let cache = crate::cache::configure_python_cache::<ChatCompletions>(
            &mut self.cache,
            py,
            arguments,
            self.call_type,
        )
        .map_err(InvokeError::Python)?;
        Ok((call, cache))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <ChatCompletions as litellm_host::protocol::Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        self.host.response(py, &response)
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
        self.host.error(py, error)
    }
    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string().into())
    }
}

impl PythonHostCalls<PythonCached<ChatCompletions>> for ChatCompletionsPythonHost {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        op: CacheCall,
    ) -> Result<(), InvokeError<Error>> {
        self.cache
            .begin(py, op)
            .map(|_| ())
            .map_err(InvokeError::Python)
    }

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        op: CacheCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        self.cache.begin(py, op).map_err(InvokeError::Python)
    }

    fn resume_host_call(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        self.cache.resume(py, result).map_err(InvokeError::Python)
    }
}

impl PythonOwned for ChatCompletionsPythonHost {
    fn close(&mut self, _: Python<'_>) {
        self.cache.close();
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.host.request)?;
        self.cache.traverse(visit)
    }
}
