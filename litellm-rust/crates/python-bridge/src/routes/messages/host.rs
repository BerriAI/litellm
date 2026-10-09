use crate::cache::{CacheCall, Cached, PythonCache, Selection};
use litellm_host_python::present;
use litellm_host_python::{PythonHostCalls, PythonOwned};

use bytes::Bytes;
use litellm_host_python::{InvokeError, PythonBinding, from_py, to_py};
use litellm_inference_messages::{
    Error, MessagesCall, MessagesSettings, MessagesShaping, messages_body,
    route::{Messages, MessagesStreamHead},
};
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities;
use pyo3::{
    exceptions::PyValueError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde_json::{Map, Value};

use crate::{
    errors::{NativeFailure, native_failure},
    marshal::{optional_timeout, project_optional_fields, public_response, request_timeout},
};

pub(super) const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.messages.route_host";

const BODY_FIELDS: [&str; 22] = [
    "max_tokens",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "inference_geo",
    "top_p",
    "mcp_servers",
    "context_management",
    "compaction",
    "container",
    "output_format",
    "speed",
    "output_config",
    "cache_control",
    "reasoning_effort",
    "safeguards",
];

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

fn merged_headers(arguments: &Bound<'_, PyDict>) -> PyResult<Option<Map<String, Value>>> {
    let mapping = |name: &str| -> PyResult<Option<Map<String, Value>>> {
        present(arguments, name)?
            .map(|value| from_py(&value))
            .transpose()
    };
    Ok(merge_headers(
        mapping("headers")?,
        mapping("extra_headers")?,
    ))
}

pub(super) struct MessagesPythonHost {
    request: Py<PyDict>,
    cache: PythonCache,
}

impl MessagesPythonHost {
    pub(super) fn new(py: Python<'_>, asynchronous: bool) -> Self {
        Self {
            request: PyDict::new(py).unbind(),
            cache: PythonCache::new(asynchronous),
        }
    }

    fn projection(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Result<MessagesCall, Error>> {
        let argument = |name: &str| present(arguments, name);
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let model = string("model")?.ok_or_else(|| PyValueError::new_err("model is required"))?;
        let messages =
            argument("messages")?.ok_or_else(|| PyValueError::new_err("messages is required"))?;
        let fields = project_optional_fields(BODY_FIELDS, argument)?;
        let body = [
            ("model".to_string(), Value::String(model.clone())),
            ("messages".to_string(), from_py(&messages)?),
        ]
        .into_iter()
        .chain(fields)
        .collect::<Map<String, Value>>();
        let timeout = request_timeout(py, arguments)?;
        let custom_llm_provider = string("custom_llm_provider")?;
        let shaping = self.shaping(py, &model, custom_llm_provider.as_deref(), arguments)?;
        let api_key = string("api_key")?;
        let api_base = string("api_base")?;
        let extra_headers = merged_headers(arguments)?;
        let provider_specific_header = present(arguments, "provider_specific_header")?
            .map(|value| from_py(&value))
            .transpose()?;
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

    fn shaping(
        &self,
        py: Python<'_>,
        model: &str,
        custom_llm_provider: Option<&str>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<MessagesShaping> {
        let module = py.import(ROUTE_HOST_MODULE)?;
        let capabilities: MessagesModelCapabilities = from_py(
            &py.import("litellm.rust_bridge.model_capabilities")?
                .getattr("anthropic_model_capabilities")?
                .call1((model, custom_llm_provider))?,
        )?;
        let settings: MessagesSettings =
            from_py(&module.getattr("settings")?.call1((arguments,))?)?;
        Ok(MessagesShaping {
            capabilities,
            settings,
        })
    }

    fn provider(&self, py: Python<'_>) -> String {
        self.request
            .bind(py)
            .get_item("custom_llm_provider")
            .ok()
            .flatten()
            .and_then(|value| value.extract::<Option<String>>().ok().flatten())
            .unwrap_or_else(|| "anthropic".into())
    }

    fn map_failure(&self, py: Python<'_>, error: PyErr) -> PyErr {
        super::super::map_failure(
            py,
            ROUTE_HOST_MODULE,
            error,
            self.request.bind(py),
            Some(&self.provider(py)),
        )
    }
}

impl PythonBinding for MessagesPythonHost {
    type Protocol = Cached<Messages>;
    type Failure = PyErr;

    fn decode_request(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<(MessagesCall, Selection), InvokeError<Error>> {
        self.request = arguments.clone().unbind();
        let selection =
            crate::cache::configure(&mut self.cache, py, arguments, "anthropic_messages")
                .map_err(InvokeError::Python)?;
        self.projection(py, arguments)
            .map_err(|error| InvokeError::Python(self.map_failure(py, error)))?
            .map_err(InvokeError::Native)
            .map(|request| (request, selection))
    }

    fn encode_response(
        &mut self,
        py: Python<'_>,
        response: Box<litellm_llms_types::formats::messages::MessagesResponse>,
    ) -> PyResult<Py<PyAny>> {
        public_response(py, ROUTE_HOST_MODULE, response.as_ref())
    }

    fn encode_stream_head(
        &mut self,
        py: Python<'_>,
        head: MessagesStreamHead,
    ) -> PyResult<Py<PyAny>> {
        py.import(ROUTE_HOST_MODULE)?
            .getattr("stream_hidden_params")?
            .call1((to_py(py, &head.headers)?,))
            .map(Bound::unbind)
    }

    fn encode_chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &chunk).into_any().unbind())
    }

    fn map_error(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        native_failure(py, NativeFailure::Messages(error), |error| {
            self.map_failure(py, error)
        })
    }

    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string().into())
    }
}

impl PythonHostCalls<Cached<Messages>> for MessagesPythonHost {
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

impl PythonOwned for MessagesPythonHost {
    fn close(&mut self, py: Python<'_>) {
        self.request = PyDict::new(py).unbind();
        self.cache.close();
    }
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)?;
        self.cache.traverse(visit)
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
}
