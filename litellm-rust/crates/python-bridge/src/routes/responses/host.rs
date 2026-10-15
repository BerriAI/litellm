use std::convert::Infallible;

use crate::{marshal::project_optional_fields, routes::inference::InferenceHost};
use litellm_auth::AwsParams;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host_python::{InvokeError, PythonBinding, PythonHostCalls, PythonOwned, from_py};
use litellm_inference_responses::{
    Error,
    ids::{self, IdScope},
    route::Responses,
    types::ResponsesCall,
};
use litellm_router_types::LitellmParams;
use pyo3::{
    exceptions::PyValueError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};

pub(super) struct ResponsesPythonHost {
    host: InferenceHost,
    ids: IdScope,
}

impl ResponsesPythonHost {
    pub fn new(host: InferenceHost) -> Self {
        Self {
            host,
            ids: IdScope::default(),
        }
    }
}

pub(super) fn project(
    host: &mut InferenceHost,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<ResponsesCall> {
    let mut call = host.project(py, arguments, "input")?;
    ids::restore_input(&mut call.input);
    let raw = |name: &str| -> PyResult<Option<String>> {
        host.argument(py, arguments, name)?
            .map(|value| value.extract())
            .transpose()
    };
    let (provider_key, provider_base): (Option<String>, Option<String>) = py
        .import(super::ROUTE_HOST_MODULE)?
        .getattr("provider_connection")?
        .call1((
            &call.options.model,
            &call.options.custom_llm_provider,
            raw("api_key")?,
            raw("api_base")?
                .filter(|base| !base.is_empty())
                .or(raw("base_url")?),
        ))?
        .extract()?;
    let litellm_params = litellm_params(
        py,
        &call.options.model,
        call.options.custom_llm_provider.as_deref(),
        project_optional_fields(
            AwsParams::fields().chain(["aws_bedrock_project_id"]),
            |name| host.argument(py, arguments, name),
        )?,
    )?;
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
        api_key: provider_key
            .filter(|key| !key.is_empty())
            .or(call.options.api_key),
        api_base: provider_base
            .filter(|base| !base.is_empty())
            .or(call.options.api_base),
        custom_llm_provider: call.options.custom_llm_provider,
        api_version: host
            .argument(py, arguments, "api_version")?
            .map(|value| value.extract())
            .transpose()?,
        litellm_params,
        extra_headers: call.options.extra_headers,
        timeout: call.options.timeout,
    })
}

/// The caller's AWS settings plus what Python reads from the model cost map for the provider
fn litellm_params(
    py: Python<'_>,
    model: &str,
    custom_llm_provider: Option<&str>,
    fields: serde_json::Map<String, serde_json::Value>,
) -> PyResult<LitellmParams> {
    let provider_params: Vec<(String, serde_json::Value)> = from_py(
        &py.import(super::ROUTE_HOST_MODULE)?
            .getattr("provider_params")?
            .call1((model, custom_llm_provider))?,
    )?;
    let fields = [(
        "model".to_owned(),
        serde_json::Value::String(model.to_owned()),
    )]
    .into_iter()
    .chain(fields)
    .chain(provider_params)
    .collect();
    serde_json::from_value(serde_json::Value::Object(fields))
        .map_err(|error| PyValueError::new_err(format!("invalid litellm params: {error}")))
}

/// The provider and deployment Python encodes into the ids it returns: the resolved provider and
/// `litellm_metadata.model_info.id`
fn id_scope(
    host: &InferenceHost,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
    call: &ResponsesCall,
) -> PyResult<IdScope> {
    let metadata = host
        .argument(py, arguments, "litellm_metadata")?
        .map(|value| from_py::<serde_json::Value>(&value))
        .transpose()?
        .unwrap_or_default();
    let model_id = match metadata.pointer("/model_info/id") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(id)) => Some(id.clone()),
        Some(other) => Some(other.to_string()),
    };
    Ok(IdScope {
        provider: get_custom_llm_provider(&call.model, call.custom_llm_provider.as_deref())
            .map(|resolved| resolved.custom_llm_provider.to_owned())
            .or_else(|| call.custom_llm_provider.clone()),
        model_id,
        encrypted_content_affinity: metadata
            .get("encrypted_content_affinity_enabled")
            .is_some_and(python_truthy),
    })
}

fn python_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(number) => number.as_f64() != Some(0.0),
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(values) => !values.is_empty(),
    }
}

impl PythonBinding for ResponsesPythonHost {
    type Protocol = Responses;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<ResponsesCall, InvokeError<Error>> {
        let call = project(&mut self.host, py, arguments).map_err(InvokeError::Python)?;
        self.ids = id_scope(&self.host, py, arguments, &call).map_err(InvokeError::Python)?;
        if call
            .optional_params
            .get("stream")
            .is_some_and(|value| value == &serde_json::Value::Bool(true))
        {
            return Err(InvokeError::Native(Error::Unsupported(
                "native Python responses streaming",
            )));
        }
        Ok(call)
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: <Responses as litellm_host::protocol::Protocol>::Response,
    ) -> PyResult<Py<PyAny>> {
        let mut response = response;
        ids::encode_response(&mut response, &self.ids);
        let response = self.host.response(py, &response)?;
        response
            .bind(py)
            .getattr("_hidden_params")?
            .set_item("custom_llm_provider", self.ids.provider.as_deref())?;
        Ok(response)
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
        self.host.error(py, error)
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
    fn close(&mut self, _: Python<'_>) {
        self.host.close();
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.host.traverse(visit)
    }
}
