use std::sync::Arc;

use litellm_core::chat_completions::chat_completions_stream as run_chat_completions_stream;
use litellm_core::chat_completions::types::{
    ChatCompletionsStreamRequest, ChatCompletionsStreamRequestBody, ChatStreamEvent,
};
use litellm_core::messages::messages_event_stream as run_messages_stream;
use litellm_core::messages::types::{
    AnthropicMessagesRequest, MessagesStreamEvent, MessagesStreamRequest,
};
use litellm_core::responses::responses_stream as run_responses_stream;
use litellm_core::responses::responses_websocket as run_responses_websocket;
use litellm_core::responses::types::{
    ResponseCommand, ResponsesStreamEvent, ResponsesStreamRequest, ResponsesStreamRequestBody,
    ResponsesWebSocketRequest,
};
use litellm_core::responses::websocket::TypedResponsesWebSocketSession;
use litellm_core::streaming::{
    JsonObject, OpenedStream, ProviderCallContext, StreamApi, StreamCapability, StreamMetadata,
    StreamProviderId, StreamTransport,
};
use litellm_python_interop::{from_py, to_py};
use pyo3::exceptions::{PyStopAsyncIteration, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::marshal::{marshal_headers, object_or_empty, optional_timeout};
use crate::routes::receiver::BridgeReceiver;
use crate::routes::runtime::{run_async, run_async_with, run_sync_with};

struct TypedEventReceiver<E> {
    metadata: StreamMetadata,
    receiver: BridgeReceiver<E>,
}

impl<E> TypedEventReceiver<E>
where
    E: Send + 'static,
{
    fn from_opened(opened: OpenedStream<E>) -> Self {
        Self {
            metadata: opened.metadata,
            receiver: BridgeReceiver::from_stream(opened.events),
        }
    }
}

fn next_event<E>(
    py: Python<'_>,
    receiver: BridgeReceiver<E>,
    stop_iteration: bool,
) -> PyResult<Py<PyAny>>
where
    E: Serialize + Send + 'static,
{
    run_sync_with(
        py,
        async move { receiver.next().await },
        move |py, event| match event {
            Some(event) => to_py(py, &event),
            None if stop_iteration => Err(PyStopIteration::new_err(())),
            None => Ok(py.None()),
        },
    )
}

fn anext_event<E>(
    py: Python<'_>,
    receiver: BridgeReceiver<E>,
    stop_iteration: bool,
) -> PyResult<Bound<'_, PyAny>>
where
    E: Serialize + Send + 'static,
{
    run_async_with(
        py,
        async move { receiver.next().await },
        move |py, event| match event {
            Some(event) => to_py(py, &event),
            None if stop_iteration => Err(PyStopAsyncIteration::new_err(())),
            None => Ok(py.None()),
        },
    )
}

macro_rules! event_stream_class {
    ($class:ident, $event:ty) => {
        #[pyclass]
        struct $class {
            inner: TypedEventReceiver<$event>,
        }

        impl From<OpenedStream<$event>> for $class {
            fn from(opened: OpenedStream<$event>) -> Self {
                Self {
                    inner: TypedEventReceiver::from_opened(opened),
                }
            }
        }

        #[pymethods]
        impl $class {
            #[getter]
            fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                to_py(py, &self.inner.metadata)
            }

            fn next_event(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                next_event(py, self.inner.receiver.clone(), false)
            }

            fn anext_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                anext_event(py, self.inner.receiver.clone(), false)
            }

            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                next_event(py, self.inner.receiver.clone(), true)
            }

            fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                anext_event(py, self.inner.receiver.clone(), true)
            }

            fn close(&self) {
                self.inner.receiver.close();
            }

            fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let receiver = self.inner.receiver.clone();
                run_async(
                    py,
                    async move {
                        receiver.close();
                        Ok(())
                    },
                    crate::errors::fallback_route_error_to_pyerr,
                )
            }
        }
    };
}

event_stream_class!(ChatCompletionsEventStream, ChatStreamEvent);
event_stream_class!(MessagesEventStream, MessagesStreamEvent);
event_stream_class!(ResponsesEventStream, ResponsesStreamEvent);

struct PythonProviderCallContext {
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
}

fn parse_call<B>(
    py: Python<'_>,
    request: Py<PyAny>,
    call: PythonProviderCallContext,
) -> PyResult<(B, ProviderCallContext)>
where
    B: DeserializeOwned,
{
    (|| -> PyResult<(B, ProviderCallContext)> {
        let body = from_py(request.bind(py))?;
        let credentials = call
            .credentials
            .map(|value| from_py(value.bind(py)))
            .transpose()?
            .unwrap_or_default();
        let extra_headers = marshal_headers(py, call.extra_headers)?
            .into_iter()
            .map(|(name, value)| litellm_core::streaming::Header { name, value })
            .collect();
        let metadata = JsonObject(object_or_empty(
            py,
            "litellm_metadata",
            call.litellm_metadata,
        )?);
        Ok((
            body,
            ProviderCallContext {
                provider: StreamProviderId::try_from(call.provider.as_str())
                    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
                credentials,
                api_base: call.api_base,
                extra_headers,
                timeout: optional_timeout(call.timeout_seconds),
                metadata,
            },
        ))
    })()
    .map_err(crate::errors::declined)
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn chat_completions_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (body, context) = parse_call::<ChatCompletionsStreamRequestBody>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_sync_with(
        py,
        async move { run_chat_completions_stream(ChatCompletionsStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, ChatCompletionsEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn achat_completions_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Bound<'_, PyAny>> {
    let (body, context) = parse_call::<ChatCompletionsStreamRequestBody>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_async_with(
        py,
        async move { run_chat_completions_stream(ChatCompletionsStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, ChatCompletionsEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn messages_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (body, context) = parse_call::<AnthropicMessagesRequest>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_sync_with(
        py,
        async move { run_messages_stream(MessagesStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, MessagesEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn amessages_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Bound<'_, PyAny>> {
    let (body, context) = parse_call::<AnthropicMessagesRequest>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_async_with(
        py,
        async move { run_messages_stream(MessagesStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, MessagesEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn responses_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (body, context) = parse_call::<ResponsesStreamRequestBody>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_sync_with(
        py,
        async move { run_responses_stream(ResponsesStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, ResponsesEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
#[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
#[allow(clippy::too_many_arguments)]
fn aresponses_stream(
    py: Python<'_>,
    request: Py<PyAny>,
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
    litellm_metadata: Option<Py<PyAny>>,
) -> PyResult<Bound<'_, PyAny>> {
    let (body, context) = parse_call::<ResponsesStreamRequestBody>(
        py,
        request,
        PythonProviderCallContext {
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        },
    )?;
    run_async_with(
        py,
        async move { run_responses_stream(ResponsesStreamRequest { body, context }).await },
        |py, opened| Ok(Py::new(py, ResponsesEventStream::from(opened))?.into_any()),
    )
}

#[pyfunction]
fn supports_streaming(api: String, provider: String, transport: String) -> PyResult<bool> {
    let capability = StreamCapability {
        api: StreamApi::try_from(api.as_str())
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        provider: StreamProviderId::try_from(provider.as_str())
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        transport: StreamTransport::try_from(transport.as_str())
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
    };
    Ok(litellm_core::streaming::supports_streaming(capability))
}

#[pyclass]
struct ResponsesWebSocketSession {
    session: Arc<dyn TypedResponsesWebSocketSession>,
}

#[pymethods]
impl ResponsesWebSocketSession {
    #[classmethod]
    #[pyo3(signature = (provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None, litellm_metadata=None))]
    #[allow(clippy::too_many_arguments)]
    fn connect<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        provider: String,
        credentials: Option<Py<PyAny>>,
        api_base: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
        litellm_metadata: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let empty_request = pyo3::types::PyDict::new(py).unbind().into_any();
        let (_, context) = parse_call::<JsonObject>(
            py,
            empty_request,
            PythonProviderCallContext {
                provider,
                credentials,
                api_base,
                extra_headers,
                timeout_seconds,
                litellm_metadata,
            },
        )?;
        run_async_with(
            py,
            async move { run_responses_websocket(ResponsesWebSocketRequest { context }).await },
            |py, session| {
                Ok(Py::new(
                    py,
                    ResponsesWebSocketSession {
                        session: Arc::from(session),
                    },
                )?
                .into_any())
            },
        )
    }

    fn send_event<'py>(&self, py: Python<'py>, command: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let command: ResponseCommand = from_py(command.bind(py))?;
        let session = self.session.clone();
        run_async(
            py,
            async move { session.send(command).await },
            crate::errors::fallback_route_error_to_pyerr,
        )
    }

    fn recv_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.clone();
        run_async_with(
            py,
            async move { session.recv().await },
            |py, event| match event {
                Some(event) => to_py(py, &event),
                None => Ok(py.None()),
            },
        )
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.clone();
        run_async(
            py,
            async move { session.close().await },
            crate::errors::fallback_route_error_to_pyerr,
        )
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChatCompletionsEventStream>()?;
    module.add_class::<MessagesEventStream>()?;
    module.add_class::<ResponsesEventStream>()?;
    module.add_class::<ResponsesWebSocketSession>()?;
    module.add_function(wrap_pyfunction!(chat_completions_stream, module)?)?;
    module.add_function(wrap_pyfunction!(achat_completions_stream, module)?)?;
    module.add_function(wrap_pyfunction!(messages_stream, module)?)?;
    module.add_function(wrap_pyfunction!(amessages_stream, module)?)?;
    module.add_function(wrap_pyfunction!(responses_stream, module)?)?;
    module.add_function(wrap_pyfunction!(aresponses_stream, module)?)?;
    module.add_function(wrap_pyfunction!(supports_streaming, module)?)
}
