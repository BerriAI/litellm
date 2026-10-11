use crate::{
    cache::{CacheCall, Cached, Selection},
    routes::inference::InferenceHost,
};
use bytes::Bytes;
use litellm_host_python::{
    InvokeError, PythonBinding, PythonHostCalls, PythonOwned, PythonTask, StreamReplay, from_py,
    to_py,
};
use litellm_inference_responses::{
    Error,
    route::Responses,
    types::{ResponsesCall, ResponsesStreamHead},
};
use litellm_llms::base_llm::responses::streaming::{
    assemble_response, decode_openai_sse, disassemble_response, encode_openai_sse,
};
use litellm_llms_types::formats::responses::ResponsesApiResponse;
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

/// The Python side of the Responses route: projects the prepared arguments and builds the
/// public response, the typed stream events and exceptions.
pub(super) struct ResponsesPythonHost {
    host: InferenceHost,
    /// The route host's event decoder for this call's provider and model, built on first use.
    decoder: Option<Py<PyAny>>,
}

impl ResponsesPythonHost {
    pub(super) fn new(host: InferenceHost) -> Self {
        Self {
            host,
            decoder: None,
        }
    }

    fn decoder<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(decoder) = &self.decoder {
            return Ok(decoder.bind(py).clone());
        }
        let decoder = py
            .import(super::ROUTE_HOST_MODULE)?
            .getattr("stream_decoder")?
            .call1((self.host.request(py)?,))?;
        self.decoder = Some(decoder.clone().unbind());
        Ok(decoder)
    }

    fn native(error: litellm_llms::Error) -> InvokeError<Error> {
        InvokeError::Native(Error::from(error))
    }
}

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
        let call = project(&mut self.host, py, arguments).map_err(InvokeError::Python)?;
        let selection = self
            .host
            .select_cache(py, arguments)
            .map_err(InvokeError::Python)?;
        Ok((call, selection))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: ResponsesApiResponse,
    ) -> PyResult<Py<PyAny>> {
        self.host.response(py, &response)
    }

    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: ResponsesStreamHead,
    ) -> PyResult<Py<PyAny>> {
        py.import(super::ROUTE_HOST_MODULE)?
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &head.headers)?,))
            .map(Bound::unbind)
    }

    /// One provider event as the typed object Python's Responses stream yields, through the
    /// provider config's own event transformation.
    fn encode_chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        let events = decode_openai_sse(&chunk).map_err(|error| {
            match self.host.error(py, Error::from(error)) {
                Ok(mapped) => mapped,
                Err(error) => error,
            }
        })?;
        let [event] = <[_; 1]>::try_from(events).map_err(|events| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "a Responses stream chunk carries one event, got {}",
                events.len()
            ))
        })?;
        let decoder = self.decoder(py)?;
        decoder.call1((to_py(py, &event)?,)).map(Bound::unbind)
    }

    fn assemble_stream(
        &mut self,
        _: Python<'_>,
        chunks: Vec<Bytes>,
    ) -> Result<ResponsesApiResponse, InvokeError<Error>> {
        let wire: Vec<u8> = chunks
            .iter()
            .flat_map(|chunk| chunk.iter().copied())
            .collect();
        let events = decode_openai_sse(&wire).map_err(Self::native)?;
        assemble_response(events)
            .map_err(Self::native)?
            .ok_or_else(|| {
                InvokeError::Native(Error::InvalidRequest(
                    "the provider stream ended without a response".into(),
                ))
            })
    }

    fn stream_of_response(
        &mut self,
        py: Python<'_>,
        response: &Py<PyAny>,
    ) -> PyResult<StreamReplay<Self::Protocol>> {
        let response: ResponsesApiResponse = from_py(response.bind(py))?;
        let public = |error: litellm_llms::Error| match self.host.error(py, Error::from(error)) {
            Ok(mapped) => mapped,
            Err(error) => error,
        };
        let chunks = disassemble_response(&response)
            .map_err(public)?
            .iter()
            .map(|event| encode_openai_sse(event).map_err(public))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(StreamReplay {
            head: ResponsesStreamHead {
                headers: Vec::new(),
            },
            chunks,
        })
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        self.host.error(py, error)
    }
    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string().into())
    }
}

impl PythonHostCalls<Cached<Responses>> for ResponsesPythonHost {
    async fn handle_host_call(
        &mut self,
        task: &PythonTask,
        op: CacheCall,
    ) -> Result<(), InvokeError<Error>> {
        self.host.cache_call(task, op).await
    }
}

impl PythonOwned for ResponsesPythonHost {
    fn close(&mut self, _: Python<'_>) {
        self.decoder = None;
        self.host.close();
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(decoder) = &self.decoder {
            visit.call(decoder)?;
        }
        self.host.traverse(visit)
    }
}
