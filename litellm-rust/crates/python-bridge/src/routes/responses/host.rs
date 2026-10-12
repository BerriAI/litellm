use crate::{
    cache::{CacheCall, Cached, Selection},
    routes::inference::InferenceHost,
};
use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned};
use litellm_inference_responses::{Error, route::Responses, types::ResponsesCall};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

pub(super) struct ResponsesPythonHost(pub InferenceHost);

pub(super) fn project(
    host: &mut InferenceHost,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<ResponsesCall> {
    let call = host.project(py, arguments, "input")?;
    let optional_params = call
        .params
        .into_iter()
        .map(|(name, value)| {
            let value = match (name.as_str(), value) {
                ("previous_response_id", serde_json::Value::String(id)) => {
                    let decoded: String = py
                        .import("litellm.responses.utils")?
                        .getattr("ResponsesAPIRequestUtils")?
                        .call_method1(
                            "decode_previous_response_id_to_original_previous_response_id",
                            (id,),
                        )?
                        .extract()?;
                    serde_json::Value::String(decoded)
                }
                (_, value) => value,
            };
            Ok((name, value))
        })
        .collect::<PyResult<_>>()?;
    Ok(ResponsesCall {
        model: call.options.model,
        input: call.input,
        optional_params,
        api_key: call.options.api_key,
        api_base: call.options.api_base,
        custom_llm_provider: call.options.custom_llm_provider,
        extra_headers: call.options.extra_headers,
        timeout: call.options.timeout,
    })
}

impl PythonBinding for ResponsesPythonHost {
    type Protocol = Cached<Responses>;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<(ResponsesCall, Selection), InvokeError<Error>> {
        let call = project(&mut self.0, py, arguments).map_err(InvokeError::Python)?;
        let selection = self
            .0
            .select_cache(py, arguments)
            .map_err(InvokeError::Python)?;
        if call
            .optional_params
            .get("stream")
            .is_some_and(|value| value == &serde_json::Value::Bool(true))
        {
            return Err(InvokeError::Native(Error::Unsupported(
                "native Python responses streaming",
            )));
        }
        Ok((call, selection))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <Cached<Responses> as litellm_host::protocol::Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        self.0.response(py, &response)
    }

    fn encode_stream_head(
        &mut self,
        _py: Python<'_>,
        head: <Cached<Responses> as litellm_host::protocol::Protocol>::StreamHead,
    ) -> PyResult<Py<PyAny>> {
        let _ = head;
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native Python Responses streaming is not supported",
        ))
    }

    fn encode_chunk(
        &mut self,
        _py: Python<'_>,
        chunk: <Cached<Responses> as litellm_host::protocol::Protocol>::Chunk,
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

impl PythonHostCalls<Cached<Responses>> for ResponsesPythonHost {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        op: CacheCall,
    ) -> Result<(), InvokeError<Error>> {
        self.0.begin_cache_call(py, op).map(|_| ())
    }

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        op: CacheCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        self.0.begin_cache_call(py, op)
    }

    fn resume_host_call(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<Error>> {
        self.0.resume_cache_call(py, result)
    }
}

impl PythonOwned for ResponsesPythonHost {
    fn close(&mut self, _: Python<'_>) {
        self.0.close();
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.traverse(visit)
    }
}
