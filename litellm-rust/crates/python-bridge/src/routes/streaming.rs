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
    JsonObject, OpenedStream, ProviderCredentials, StreamMetadata, StreamProviderId, StreamTarget,
    StreamTransportOptions,
};
use litellm_python_interop::{from_py, to_py};
use pyo3::exceptions::{PyStopAsyncIteration, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::marshal::{marshal_headers, optional_timeout};
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
        crate::errors::executed_route_error_to_pyerr,
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
        crate::errors::executed_route_error_to_pyerr,
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
                    crate::errors::executed_route_error_to_pyerr,
                )
            }
        }
    };
}

event_stream_class!(ChatCompletionsEventStream, ChatStreamEvent);
event_stream_class!(MessagesEventStream, MessagesStreamEvent);
event_stream_class!(ResponsesEventStream, ResponsesStreamEvent);

#[derive(Default, Deserialize)]
struct PythonProviderCredentials {
    api_key: Option<String>,
    aws_access_key_id: Option<String>,
    aws_secret_access_key: Option<String>,
    aws_session_token: Option<String>,
}

struct PythonStreamTarget {
    provider: String,
    credentials: Option<Py<PyAny>>,
    api_base: Option<String>,
}

struct PythonStreamTransport {
    extra_headers: Option<Py<PyAny>>,
    timeout_seconds: Option<f64>,
}

fn parse_call<B>(
    py: Python<'_>,
    request: Py<PyAny>,
    target: PythonStreamTarget,
    transport: PythonStreamTransport,
) -> PyResult<(B, StreamTarget, StreamTransportOptions)>
where
    B: DeserializeOwned,
{
    (|| -> PyResult<(B, StreamTarget, StreamTransportOptions)> {
        let body = from_py(request.bind(py))?;
        let credentials = target
            .credentials
            .map(|value| from_py::<PythonProviderCredentials>(value.bind(py)))
            .transpose()?
            .unwrap_or_default();
        let provider = StreamProviderId::try_from(target.provider.as_str())
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let target = StreamTarget::new(
            provider,
            ProviderCredentials::new(
                credentials.api_key,
                credentials.aws_access_key_id,
                credentials.aws_secret_access_key,
                credentials.aws_session_token,
            ),
            target.api_base,
        );
        let extra_headers = transport
            .extra_headers
            .map(|value| from_py(value.bind(py)))
            .transpose()?;
        let forwarded_headers = marshal_headers(extra_headers)?
            .into_iter()
            .map(|(name, value)| litellm_core::streaming::Header { name, value })
            .collect();
        let transport = StreamTransportOptions::new(
            forwarded_headers,
            optional_timeout(transport.timeout_seconds),
        );
        Ok((body, target, transport))
    })()
    .map_err(crate::errors::declined)
}

macro_rules! stream_openers {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        body = $body:ty,
        request = $request:ident,
        open = $open:path,
        stream = $stream:ident
    ) => {
        #[pyfunction]
        #[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None))]
        #[allow(clippy::too_many_arguments)]
        fn $sync_name(
            py: Python<'_>,
            request: Py<PyAny>,
            provider: String,
            credentials: Option<Py<PyAny>>,
            api_base: Option<String>,
            extra_headers: Option<Py<PyAny>>,
            timeout_seconds: Option<f64>,
        ) -> PyResult<Py<PyAny>> {
            let (body, target, transport) = parse_call::<$body>(
                py,
                request,
                PythonStreamTarget {
                    provider,
                    credentials,
                    api_base,
                },
                PythonStreamTransport {
                    extra_headers,
                    timeout_seconds,
                },
            )?;
            run_sync_with(
                py,
                async move { $open($request { body, target, transport }).await },
                crate::errors::fallback_route_error_to_pyerr,
                |py, opened| Ok(Py::new(py, $stream::from(opened))?.into_any()),
            )
        }

        #[pyfunction]
        #[pyo3(signature = (request, provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: Python<'_>,
            request: Py<PyAny>,
            provider: String,
            credentials: Option<Py<PyAny>>,
            api_base: Option<String>,
            extra_headers: Option<Py<PyAny>>,
            timeout_seconds: Option<f64>,
        ) -> PyResult<Bound<'_, PyAny>> {
            let (body, target, transport) = parse_call::<$body>(
                py,
                request,
                PythonStreamTarget {
                    provider,
                    credentials,
                    api_base,
                },
                PythonStreamTransport {
                    extra_headers,
                    timeout_seconds,
                },
            )?;
            run_async_with(
                py,
                async move { $open($request { body, target, transport }).await },
                crate::errors::fallback_route_error_to_pyerr,
                |py, opened| Ok(Py::new(py, $stream::from(opened))?.into_any()),
            )
        }
    };
}

stream_openers! {
    sync = chat_completions_stream,
    asynchronous = achat_completions_stream,
    body = ChatCompletionsStreamRequestBody,
    request = ChatCompletionsStreamRequest,
    open = run_chat_completions_stream,
    stream = ChatCompletionsEventStream
}

stream_openers! {
    sync = messages_stream,
    asynchronous = amessages_stream,
    body = AnthropicMessagesRequest,
    request = MessagesStreamRequest,
    open = run_messages_stream,
    stream = MessagesEventStream
}

stream_openers! {
    sync = responses_stream,
    asynchronous = aresponses_stream,
    body = ResponsesStreamRequestBody,
    request = ResponsesStreamRequest,
    open = run_responses_stream,
    stream = ResponsesEventStream
}

#[pyclass]
struct ResponsesWebSocketSession {
    session: Arc<dyn TypedResponsesWebSocketSession>,
}

#[pymethods]
impl ResponsesWebSocketSession {
    #[classmethod]
    #[pyo3(signature = (provider, credentials=None, api_base=None, extra_headers=None, timeout_seconds=None))]
    #[allow(clippy::too_many_arguments)]
    fn connect<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        provider: String,
        credentials: Option<Py<PyAny>>,
        api_base: Option<String>,
        extra_headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let empty_request = pyo3::types::PyDict::new(py).unbind().into_any();
        let (_, target, transport) = parse_call::<JsonObject>(
            py,
            empty_request,
            PythonStreamTarget {
                provider,
                credentials,
                api_base,
            },
            PythonStreamTransport {
                extra_headers,
                timeout_seconds,
            },
        )?;
        run_async_with(
            py,
            async move {
                run_responses_websocket(ResponsesWebSocketRequest { target, transport }).await
            },
            crate::errors::fallback_route_error_to_pyerr,
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
            crate::errors::executed_route_error_to_pyerr,
        )
    }

    fn recv_event<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.clone();
        run_async_with(
            py,
            async move { session.recv().await },
            crate::errors::executed_route_error_to_pyerr,
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
            crate::errors::executed_route_error_to_pyerr,
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
    module.add_function(wrap_pyfunction!(aresponses_stream, module)?)
}
