use bytes::Bytes;
use litellm_core::messages::{
    Error,
    route::{Messages, MessagesCall, MessagesOp, MessagesOpResult, MessagesOutput},
};
use litellm_host_python::{
    Completed, Invoke, InvokeError, ResponseOrigin, RouteHost, from_py, lookup, to_py,
};
use litellm_http::transport::Error as TransportError;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use pyo3::{
    exceptions::{PyException, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde_json::{Map, Value};
use std::time::Instant;

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

/// A route operation the host answered with a Python awaitable.
enum Pending {
    Lookup {
        arguments: Py<PyDict>,
        started: Instant,
    },
}

/// The Python side of the Messages route: projects the prepared arguments, serves the
/// response cache, and builds the public response, chunks and exceptions.
pub(super) struct MessagesRouteHost {
    request: Py<PyAny>,
    pending: Option<Pending>,
    call_type: &'static str,
    asynchronous: bool,
}

impl MessagesRouteHost {
    pub(super) fn new(request: Py<PyAny>, asynchronous: bool) -> Self {
        Self {
            request,
            pending: None,
            call_type: if asynchronous {
                "aanthropic_messages"
            } else {
                "anthropic_messages"
            },
            asynchronous,
        }
    }

    fn public_response(
        &self,
        py: Python<'_>,
        message: &AnthropicMessagesResponse,
        origin: ResponseOrigin,
    ) -> PyResult<Completed> {
        py.import("litellm.rust_bridge.messages.route_host")?
            .getattr("response")?
            .call1((to_py(py, message)?,))
            .map(|response| Completed {
                response: response.unbind(),
                origin,
            })
    }

    fn lookup_cache(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> Result<Invoke<Messages>, InvokeError<Error>> {
        let started = Instant::now();
        let module = py
            .import("litellm.rust_bridge.call_cache")
            .map_err(|error| InvokeError::Python(self.map_failure(py, error)))?;
        if self.asynchronous {
            let awaitable = module
                .getattr("lookup")
                .and_then(|lookup| lookup.call1((self.call_type, arguments)))
                .map(Bound::unbind)
                .map_err(|error| InvokeError::Python(self.map_failure(py, error)))?;
            self.pending = Some(Pending::Lookup {
                arguments: arguments.clone().unbind(),
                started,
            });
            Ok(Invoke::Await(awaitable))
        } else {
            module
                .getattr("lookup_sync")
                .and_then(|lookup| lookup.call1((self.call_type, arguments)))
                .and_then(|result| self.cached(py, &result, arguments, started))
                .map(Invoke::Ready)
                .map_err(|error| InvokeError::Python(self.map_failure(py, error)))
        }
    }

    fn cached(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        arguments: &Bound<'_, PyDict>,
        started: Instant,
    ) -> PyResult<MessagesOpResult> {
        if value.is_none() {
            return Ok(MessagesOpResult::Cached(None));
        }
        match from_py::<AnthropicMessagesResponse>(value) {
            Ok(message) => {
                py.import("litellm.rust_bridge.call_cache")?
                    .getattr("mark_hit")?
                    .call1((
                        self.call_type,
                        arguments,
                        value,
                        self.asynchronous,
                        started.elapsed().as_secs_f64() * 1000.0,
                    ))?;
                Ok(MessagesOpResult::Cached(Some(Box::new(message))))
            }
            Err(_) => Ok(MessagesOpResult::Cached(None)),
        }
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
    ) -> Result<Invoke<Messages>, InvokeError<Error>> {
        match op {
            MessagesOp::ProjectRequest => self
                .project(py, arguments)
                .map(|call| MessagesOpResult::Request(Box::new(call)))
                .map(Invoke::Ready)
                .map_err(|error| InvokeError::Python(self.map_failure(py, error))),
            MessagesOp::LookupCache => self.lookup_cache(py, arguments),
            MessagesOp::StoreCache(message) => {
                let store = if self.asynchronous {
                    "store"
                } else {
                    "store_sync"
                };
                py.import("litellm.rust_bridge.call_cache")
                    .and_then(|module| module.getattr(store))
                    .and_then(|call| {
                        call.call1((self.call_type, arguments, to_py(py, message.as_ref())?))
                    })
                    .map(|_| Invoke::Ready(MessagesOpResult::Stored))
                    .map_err(|error| InvokeError::Python(self.map_failure(py, error)))
            }
        }
    }

    fn resume_op(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<MessagesOpResult, InvokeError<Error>> {
        match self.pending.take() {
            Some(Pending::Lookup { arguments, started }) => {
                let value =
                    result.map_err(|error| InvokeError::Python(self.map_failure(py, error)))?;
                self.cached(py, value.bind(py), arguments.bind(py), started)
                    .map_err(|error| InvokeError::Python(self.map_failure(py, error)))
            }
            None => Err(InvokeError::Python(
                pyo3::exceptions::PyRuntimeError::new_err("route host has no pending operation"),
            )),
        }
    }

    fn complete(&mut self, py: Python<'_>, response: MessagesOutput) -> PyResult<Completed> {
        match response {
            MessagesOutput::Message(message) => {
                self.public_response(py, message.as_ref(), ResponseOrigin::Provider)
            }
            MessagesOutput::Cached(message) => {
                self.public_response(py, message.as_ref(), ResponseOrigin::Cache)
            }
            MessagesOutput::Streamed => Ok(Completed {
                response: py.None(),
                origin: ResponseOrigin::Provider,
            }),
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
