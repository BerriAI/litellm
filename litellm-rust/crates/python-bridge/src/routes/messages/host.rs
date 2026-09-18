use bytes::Bytes;
use litellm_core::messages::{
    Error,
    route::{Messages, MessagesCall, MessagesOp, MessagesOpResult, MessagesOutput},
};
use litellm_host_python::{InvokeError, RouteHost, from_py, lookup, to_py};
use litellm_llms::custom_httpx::transport::Error as TransportError;
use pyo3::{
    exceptions::{PyException, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde_json::{Map, Value};

use crate::{
    errors::{RustUpstreamError, messages_error_to_pyerr},
    marshal::{optional_timeout, python_timeout_seconds},
};

/// The Anthropic Messages body fields a caller may pass besides `model` and `messages`,
/// as `AnthropicMessagesRequestOptionalParams` declares them.
const BODY_FIELDS: [&str; 20] = [
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
    "container",
    "output_format",
    "speed",
    "output_config",
    "cache_control",
    "reasoning_effort",
];

/// The Python side of the Messages route: projects the prepared arguments and builds the
/// public response, chunks and exceptions.
pub(super) struct MessagesRouteHost {
    request: Py<PyAny>,
}

impl MessagesRouteHost {
    pub(super) fn new(request: Py<PyAny>) -> Self {
        Self { request }
    }

    fn project(&self, py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<MessagesCall> {
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
        Ok(MessagesCall {
            model,
            body,
            api_key: string("api_key")?,
            api_base: string("api_base")?,
            custom_llm_provider: string("custom_llm_provider")?,
            extra_headers: argument("extra_headers")?
                .map(|value| from_py(&value))
                .transpose()?,
            timeout: optional_timeout(timeout),
        })
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
            .import("litellm.rust_bridge.messages.route_host")
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

impl RouteHost for MessagesRouteHost {
    type Route = Messages;
    type Failure = PyErr;

    fn invoke(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        op: MessagesOp,
    ) -> Result<MessagesOpResult, InvokeError<Error>> {
        match op {
            MessagesOp::ProjectRequest => self
                .project(py, arguments)
                .map(|call| MessagesOpResult::Request(Box::new(call)))
                .map_err(|error| InvokeError::Python(self.map_failure(py, error))),
        }
    }

    fn complete(&mut self, py: Python<'_>, response: MessagesOutput) -> PyResult<Py<PyAny>> {
        match response {
            MessagesOutput::Message(message) => py
                .import("litellm.rust_bridge.messages.route_host")?
                .getattr("response")?
                .call1((to_py(py, message.as_ref())?,))
                .map(Bound::unbind),
            MessagesOutput::Streamed => Ok(py.None()),
        }
    }

    fn chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &chunk).into_any().unbind())
    }

    fn classify(&self, py: Python<'_>, error: Error) -> PyResult<PyErr> {
        let native = match error {
            Error::Transport(TransportError::Http { status, body }) => {
                let error = RustUpstreamError::new_err((status, body));
                error
                    .value(py)
                    .setattr("headers", Vec::<(String, String)>::new())?;
                error
            }
            other => messages_error_to_pyerr(other),
        };
        Ok(self.map_failure(py, native))
    }

    fn host_error(error: &PyErr) -> Error {
        Error::InvalidRequest(error.to_string())
    }

    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.request)
    }
}
