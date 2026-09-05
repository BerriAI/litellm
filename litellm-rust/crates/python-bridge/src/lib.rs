mod diagnostics;
mod errors;
mod execution;
#[cfg(feature = "trace-parity")]
mod function_trace;
mod marshal;
mod routes;

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use litellm_core::responses::types::ResponsesWebSocketRequest;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions};

#[derive(FromPyObject)]
struct WebSocketConnectRequest {
    url: String,
}

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustResponsesWebSocketConnection,
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (request, *, options, context))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        request: WebSocketConnectRequest,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(reason) = responses_websocket_decline(
            "responses websocket",
            options.provider("openai"),
            false,
            false,
            false,
            None,
        ) {
            return Err(crate::errors::RustBridgeDeclined::new_err(reason));
        }
        let options: litellm_core::request_options::RequestOptions = options.into();
        let context: litellm_core::request_context::LiteLlmRequestContext = context.into();
        let request = ResponsesWebSocketRequest { url: request.url };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustResponsesWebSocketConnection::connect(request, &options, &context)
                .await
                .map_err(core_error_to_pyerr)?;
            Ok(ResponsesWebSocketConnection { inner })
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.recv_text().await.map_err(core_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_pyerr)
        })
    }
}

#[pyfunction]
#[pyo3(signature = (_model, custom_llm_provider, *, stream=false, has_agentic_hook=false, has_custom_client=false, request_format=None))]
fn responses_websocket_decline(
    _model: &str,
    custom_llm_provider: &str,
    stream: bool,
    has_agentic_hook: bool,
    has_custom_client: bool,
    request_format: Option<&str>,
) -> Option<String> {
    routes::definition::request_decline(
        litellm_core::responses::websocket::native_websocket_supported(custom_llm_provider),
        stream,
        has_agentic_hook,
        has_custom_client,
        request_format,
    )
}

#[pymodule(gil_used = false)]
mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        module.add_class::<super::ResponsesWebSocketConnection>()?;
        module.add_function(wrap_pyfunction!(
            super::responses_websocket_decline,
            module
        )?)?;
        super::diagnostics::register(module)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);

            let expected = [
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ocr_decline",
                "ocr",
                "aocr",
                "transcription_decline",
                "transcription",
                "atranscription",
                "messages_decline",
                "messages",
                "amessages",
                "chat_completions_decline",
                "chat_completions",
                "achat_completions",
                "ResponsesWebSocketConnection",
                "responses_websocket_decline",
                "gil_stats",
            ];

            let public_names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            assert_eq!(public_names, expected);

            #[cfg(not(feature = "trace-parity"))]
            assert!(!module.hasattr("_trace").expect("module lookup should work"));

            #[cfg(feature = "trace-parity")]
            {
                let trace = module
                    .getattr("_trace")
                    .expect("trace build should expose its diagnostic namespace");
                let trace_names: Vec<String> = trace
                    .cast::<PyModule>()
                    .expect("trace namespace should be a module")
                    .dict()
                    .keys()
                    .extract::<Vec<String>>()
                    .expect("trace names should be strings")
                    .into_iter()
                    .filter(|name| !name.starts_with("__"))
                    .collect();
                assert_eq!(
                    trace_names,
                    [
                        "ocr",
                        "aocr",
                        "transcription",
                        "atranscription",
                        "messages",
                        "amessages",
                        "chat_completions",
                        "achat_completions",
                        "gateway_messages",
                    ]
                );
            }
        });
    }

    #[test]
    fn responses_websocket_connection_round_trips_through_python() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        let server = runtime.spawn(async move {
            let (stream, _) = listener.accept().await.expect("server should accept");
            let mut socket = accept_async(stream)
                .await
                .expect("handshake should succeed");

            let message = socket
                .next()
                .await
                .expect("client should send a frame")
                .expect("client frame should be valid");
            assert_eq!(message, Message::Text("from-python".into()));
            socket
                .send(Message::Text("from-server".into()))
                .await
                .expect("server should reply");
            assert!(matches!(socket.next().await, Some(Ok(Message::Close(_)))));
        });

        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);
            let locals = crate::marshal::request_fixtures(py);
            locals
                .set_item("native", &module)
                .expect("module should enter Python locals");
            locals
                .set_item("url", format!("ws://{address}"))
                .expect("URL should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    for request, request_options, request_context, field in (
        (Request(url=123), options, context, 'url'),
        (Request(url=url), Options(extra_headers=[]), context, 'extra_headers'),
        (Request(url=url), options, replace(context, litellm_call_id=123), 'litellm_call_id'),
        (Request(url=url), options, replace(context, attribution=Attribution(user_api_key_user_id=123)), 'user_api_key_user_id'),
    ):
        try:
            native.ResponsesWebSocketConnection.connect(request, options=request_options, context=request_context)
        except (TypeError, ValueError) as error:
            parts = []
            while error is not None:
                parts.append(str(error))
                error = error.__cause__
            assert field in ' / '.join(parts), parts
        else:
            raise AssertionError('invalid WebSocket input reached execution')

    connection = await native.ResponsesWebSocketConnection.connect(Request(url=url), options=options, context=context)
    assert type(connection) is native.ResponsesWebSocketConnection
    await connection.send_text("from-python")
    assert await connection.recv_text() == "from-server"
    await connection.close()
    assert await connection.recv_text() is None

asyncio.run(asyncio.wait_for(exercise(), timeout=5))
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("Python WebSocket methods should round trip");
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }
}
