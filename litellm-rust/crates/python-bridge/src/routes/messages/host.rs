use std::convert::Infallible;

use bytes::Bytes;
use litellm_core::messages::{
    Error, MessagesCall, MessagesShaping, messages_body,
    route::{BODY_FIELDS, Messages, MessagesOutput, MessagesStreamHead},
};
use litellm_host_python::{InvokeError, ProtocolHost, from_py, lookup, to_py};
use litellm_http::transport::Error as TransportError;
use litellm_types::utils::ProviderSpecificHeaders;
use pyo3::{
    exceptions::{PyException, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};

use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{optional_timeout, python_timeout_seconds},
};

const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.messages.route_host";
const REQUEST_ERROR_MARKER: &str = "messages_request_error";

fn merge_headers(
    forwarded: Option<Map<String, Value>>,
    extra_headers: Option<Map<String, Value>>,
) -> Option<Map<String, Value>> {
    let merged: Map<String, Value> = forwarded
        .into_iter()
        .flatten()
        .chain(extra_headers.into_iter().flatten())
        .collect();
    (!merged.is_empty()).then_some(merged)
}

fn header_pairs(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect()
}

fn native_error(py: Python<'_>, error: Error) -> PyResult<PyErr> {
    match error {
        Error::Transport(TransportError::Http { status, body }) => {
            let error = RustUpstreamError::new_err((status, body));
            error
                .value(py)
                .setattr("headers", Vec::<(String, String)>::new())?;
            Ok(error)
        }
        Error::InvalidRequest(message) => {
            let error = PyValueError::new_err(message);
            error.value(py).setattr(REQUEST_ERROR_MARKER, true)?;
            Ok(error)
        }
        Error::MissingField(field) => {
            let error = PyValueError::new_err(format!("missing required field: {field}"));
            error.value(py).setattr(REQUEST_ERROR_MARKER, true)?;
            Ok(error)
        }
        other => Ok(route_error_to_pyerr(other)),
    }
}

/// The Python side of the Messages route: projects the prepared arguments and builds the
/// public response, chunks and exceptions.
pub(super) struct MessagesPythonHost {
    request: Py<PyAny>,
}

impl MessagesPythonHost {
    pub(super) fn new(request: Py<PyAny>) -> Self {
        Self { request }
    }

    fn projection(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Result<MessagesCall, Error>> {
        let request = self.request.bind(py);
        let argument = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(lookup(arguments, request, name)?.filter(|value| !value.is_none()))
        };
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let model = string("model")?.ok_or_else(|| PyValueError::new_err("model is required"))?;
        let messages =
            argument("messages")?.ok_or_else(|| PyValueError::new_err("messages is required"))?;
        let fields = BODY_FIELDS
            .iter()
            .filter(|name| **name != "messages")
            .filter_map(|name| match argument(name) {
                Ok(Some(value)) => Some(from_py(&value).map(|value| ((*name).to_string(), value))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<PyResult<Vec<(String, Value)>>>()?;
        let body = [
            ("model".to_string(), Value::String(model.clone())),
            ("messages".to_string(), from_py(&messages)?),
        ]
        .into_iter()
        .chain(fields)
        .collect::<Map<String, Value>>();
        let timeout = argument("timeout")?
            .map(|value| python_timeout_seconds(py, value.unbind()))
            .transpose()?
            .flatten();
        let custom_llm_provider = string("custom_llm_provider")?;
        let shaping = self.shaping(py, &model, custom_llm_provider.as_deref(), arguments)?;
        let api_key = string("api_key")?;
        let api_base = string("api_base")?;
        let extra_headers = self.merged_headers(py, arguments)?;
        let provider_specific_header = self.provider_specific_header(py, arguments)?;
        Ok(messages_body(body).map(|body| MessagesCall {
            body,
            api_key,
            api_base,
            extra_headers,
            provider_specific_header,
            custom_llm_provider,
            timeout: optional_timeout(timeout),
            shaping,
        }))
    }

    fn merged_headers(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Map<String, Value>>> {
        let request = self.request.bind(py);
        let mapping = |name: &str| -> PyResult<Option<Map<String, Value>>> {
            lookup(arguments, request, name)?
                .filter(|value| !value.is_none())
                .map(|value| from_py(&value))
                .transpose()
        };
        Ok(merge_headers(
            mapping("headers")?,
            mapping("extra_headers")?,
        ))
    }

    fn provider_specific_header(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Option<ProviderSpecificHeaders>> {
        lookup(arguments, self.request.bind(py), "provider_specific_header")?
            .filter(|value| !value.is_none())
            .map(|value| from_py(&value))
            .transpose()
    }

    fn shaping(
        &self,
        py: Python<'_>,
        model: &str,
        custom_llm_provider: Option<&str>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<MessagesShaping> {
        let projected = py.import(ROUTE_HOST_MODULE)?.getattr("shaping")?.call1((
            model,
            custom_llm_provider,
            arguments,
        ))?;
        from_py(&projected)
    }

    fn provider(&self, py: Python<'_>) -> String {
        self.request
            .bind(py)
            .getattr("custom_llm_provider")
            .and_then(|value| value.extract::<Option<String>>())
            .ok()
            .flatten()
            .unwrap_or_else(|| "anthropic".into())
    }

    fn map_failure(&self, py: Python<'_>, error: PyErr) -> PyErr {
        if !error.is_instance_of::<PyException>(py) {
            return error;
        }
        let mapped = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.getattr("map_failure"))
            .and_then(|map| map.call1((error.value(py), self.request.bind(py), self.provider(py))))
            .and_then(|mapped| {
                mapped
                    .extract::<Py<pyo3::exceptions::PyBaseException>>()
                    .map_err(PyErr::from)
            });
        match mapped {
            Ok(mapped) => PyErr::from_value(mapped.into_bound(py).into_any()),
            Err(_) => error,
        }
    }
}

impl ProtocolHost for MessagesPythonHost {
    type Protocol = Messages;
    type Failure = PyErr;

    fn project(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<MessagesCall, InvokeError<Error>> {
        self.projection(py, arguments)
            .map_err(|error| InvokeError::Python(self.map_failure(py, error)))?
            .map_err(InvokeError::Native)
    }

    fn invoke(&mut self, _: Python<'_>, op: Infallible) -> Result<(), InvokeError<Error>> {
        match op {}
    }

    fn complete(&mut self, py: Python<'_>, response: MessagesOutput) -> PyResult<Py<PyAny>> {
        match response {
            MessagesOutput::Message(message) => py
                .import(ROUTE_HOST_MODULE)?
                .getattr("response")?
                .call1((to_py(py, message.as_ref())?,))
                .map(Bound::unbind),
            MessagesOutput::Streamed => Ok(py.None()),
        }
    }

    fn head(&mut self, py: Python<'_>, head: MessagesStreamHead) -> PyResult<Py<PyAny>> {
        py.import(ROUTE_HOST_MODULE)?
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &header_pairs(&head.headers))?,))
            .map(Bound::unbind)
    }

    fn chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &chunk).into_any().unbind())
    }

    fn classify(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        if let Error::Secret(source) = &error
            && let Some(original) = crate::secrets::python_error(py, source.source_error())
        {
            return Ok(original);
        }
        Ok(self.map_failure(py, native_error(py, error)?))
    }

    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string())
    }

    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn map(value: Value) -> Map<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::abandoned(litellm_host::MachineFault::Abandoned)]
    #[case::unsupported(litellm_host::MachineFault::Unsupported("streaming"))]
    fn host_faults_are_internal_errors(#[case] fault: litellm_host::MachineFault) {
        Python::initialize();
        Python::attach(|py| {
            let error = Error::from(fault);
            assert!(!error.is_request());
            let mapped = native_error(py, error).unwrap();
            assert!(mapped.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(!mapped.value(py).hasattr(REQUEST_ERROR_MARKER).unwrap());
        });
    }

    #[rstest]
    #[case::extra_over_forwarded(
        Some(json!({"X-Priority": "forwarded", "X-Forwarded-Only": "keep"})),
        Some(json!({"X-Priority": "extra", "X-Extra-Only": "also-keep"})),
        Some(json!({"X-Priority": "extra", "X-Forwarded-Only": "keep", "X-Extra-Only": "also-keep"})),
    )]
    #[case::only_forwarded(Some(json!({"X-Forwarded": "yes"})), None, Some(json!({"X-Forwarded": "yes"})))]
    #[case::only_extra_headers(
        None,
        Some(json!({"X-Custom-Header": "from-kwargs", "X-Auth-Token": "token123"})),
        Some(json!({"X-Custom-Header": "from-kwargs", "X-Auth-Token": "token123"})),
    )]
    #[case::nothing(None, Some(json!({})), None)]
    fn headers_merge_forwarded_then_extra(
        #[case] forwarded: Option<Value>,
        #[case] extra_headers: Option<Value>,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            merge_headers(forwarded.map(map), extra_headers.map(map)),
            expected.map(map)
        );
    }

    #[test]
    fn header_pairs_drop_opaque_values_and_keep_duplicates() {
        let mut headers = HeaderMap::new();
        headers.append("X-Multi", "a".parse().unwrap());
        headers.append("X-Multi", "b".parse().unwrap());
        headers.append(
            "x-opaque",
            reqwest::header::HeaderValue::from_bytes(&[0xff]).unwrap(),
        );
        assert_eq!(
            header_pairs(&headers),
            vec![
                ("x-multi".to_string(), "a".to_string()),
                ("x-multi".to_string(), "b".to_string())
            ]
        );
    }

    #[rstest]
    #[case::rejected_request(Error::InvalidRequest("does not support top_k=5".into()), true)]
    #[case::missing_field(Error::MissingField("max_tokens"), true)]
    #[case::request_decoding(
        Error::RequestDecoding(serde_json::from_value::<()>(json!("x")).unwrap_err().into()),
        true,
    )]
    #[case::response_decoding(
        Error::ResponseDecoding(serde_json::from_value::<()>(json!("x")).unwrap_err().into()),
        false,
    )]
    #[case::unresolvable_provider(Error::InvalidProvider("openai".into()), false)]
    #[case::upstream_failure(
        Error::Transport(TransportError::Http { status: 400, body: "bad".into() }),
        false,
    )]
    fn only_request_rejections_carry_the_request_error_marker(
        #[case] error: Error,
        #[case] marked: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let native = native_error(py, error).unwrap();
            let marker = native
                .value(py)
                .getattr_opt(REQUEST_ERROR_MARKER)
                .unwrap()
                .map(|value| value.extract::<bool>().unwrap());
            assert_eq!(marker.unwrap_or(false), marked);
        });
    }
}
