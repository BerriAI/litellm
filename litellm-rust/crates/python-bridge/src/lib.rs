mod callback_bindings;
#[cfg(test)]
#[path = "../tests/callbacks/mod.rs"]
mod callback_tests;
mod constants;
mod diagnostics;
mod errors;
mod execution;
#[cfg(feature = "trace-parity")]
mod function_trace;
mod marshal;
mod python_hook_bindings;
mod routes;

use std::sync::atomic::{AtomicU64, Ordering};

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use litellm_core::provider_callbacks::{CallbackDecision, SessionEvent, SessionObserver};
use litellm_core::responses::types::ResponsesWebSocketRequest;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions};

static NEXT_WEBSOCKET_SESSION_ID: AtomicU64 = AtomicU64::new(1);

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
    #[pyo3(signature = (request, *, options, context, callback_adapter=None))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        request: WebSocketConnectRequest,
        options: NativeRequestOptions,
        context: NativeRequestContext,
        callback_adapter: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider_supported = litellm_core::responses::websocket::native_websocket_supported(
            options.provider("openai"),
        );
        let context: litellm_core::request_context::LiteLlmRequestContext = context.into();
        if let Some(reason) = routes::definition::request_decline(provider_supported, &context) {
            return Err(crate::errors::RustBridgeDeclined::new_err(reason));
        }
        let options: litellm_core::request_options::RequestOptions = options.into();
        let call_id = context.litellm_call_id.clone().unwrap_or_default();
        let session_id = format!(
            "responses-websocket-{}",
            NEXT_WEBSOCKET_SESSION_ID.fetch_add(1, Ordering::Relaxed)
        );
        let mut observer = callback_adapter
            .map(|adapter| crate::callback_bindings::python_async_session(adapter, py))
            .transpose()?;
        let request = ResponsesWebSocketRequest { url: request.url };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            if let Some(observer) = observer.as_mut() {
                let decision = observer
                    .before_connect(&session_event(&session_id, &call_id, None))
                    .await?;
                match decision {
                    CallbackDecision::Unchanged => {}
                    CallbackDecision::Replace { .. } => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "before_connect cannot replace WebSocket setup",
                        ));
                    }
                    CallbackDecision::Reject { message, .. } => {
                        return Err(pyo3::exceptions::PyValueError::new_err(message));
                    }
                }
            }
            let inner = match RustResponsesWebSocketConnection::connect(request, &options, &context).await {
                Ok(inner) => inner,
                Err(error) => {
                    if let Some(observer) = observer.as_mut() {
                        observer
                            .error(&session_event(
                                &session_id,
                                &call_id,
                                Some(error.to_string()),
                            ))
                            .await?;
                    }
                    return Err(core_error_to_pyerr(error));
                }
            };
            if let Some(observer) = observer.as_mut() {
                observer
                    .connected(&session_event(&session_id, &call_id, None))
                    .await?;
            }
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

fn session_event(session_id: &str, call_id: &str, message: Option<String>) -> SessionEvent {
    SessionEvent {
        session_id: session_id.to_string(),
        call_id: call_id.to_string(),
        trace_id: None,
        event: None,
        response_id: None,
        sequence: None,
        message,
    }
}

#[pyfunction]
#[pyo3(signature = (_model, custom_llm_provider, *, context))]
fn responses_websocket_decline(
    _model: &str,
    custom_llm_provider: &str,
    context: NativeRequestContext,
) -> Option<String> {
    let context: litellm_core::request_context::LiteLlmRequestContext = context.into();
    routes::definition::request_decline(
        litellm_core::responses::websocket::native_websocket_supported(custom_llm_provider),
        &context,
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
import sys
import types

litellm_module = types.ModuleType('litellm')
rust_bridge_module = types.ModuleType('litellm.rust_bridge')
litellm_module.rust_bridge = rust_bridge_module
rust_bridge_module._native = native
sys.modules['litellm'] = litellm_module
sys.modules['litellm.rust_bridge'] = rust_bridge_module
sys.modules['litellm.rust_bridge._native'] = native

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

    events = []

    class Adapter:
        async def before_connect(self, event):
            events.append(('before_connect', event))
            return {'action': 'unchanged'}

        async def connected(self, event):
            events.append(('connected', event))

        async def before_send(self, event):
            return {'action': 'unchanged'}

        async def after_receive(self, event):
            return {'action': 'unchanged'}

        async def response_complete(self, event):
            pass

        async def response_error(self, event):
            pass

        async def error(self, event):
            events.append(('error', event))

        async def close(self, event):
            pass

    class RejectingAdapter(Adapter):
        async def before_connect(self, event):
            return {'action': 'reject', 'message': 'blocked', 'status_code': 400}

    try:
        await native.ResponsesWebSocketConnection.connect(
            Request(url=url), options=options, context=context, callback_adapter=RejectingAdapter()
        )
    except ValueError as error:
        assert str(error) == 'blocked'
    else:
        raise AssertionError('rejected WebSocket setup reached execution')

    connection = await native.ResponsesWebSocketConnection.connect(
        Request(url=url),
        options=options,
        context=replace(context, litellm_call_id='call-1'),
        callback_adapter=Adapter(),
    )
    assert [name for name, _ in events] == ['before_connect', 'connected']
    assert events[0][1]['call_id'] == 'call-1'
    assert events[0][1]['session_id'] == events[1][1]['session_id']
    assert type(connection) is native.ResponsesWebSocketConnection
    await connection.send_text("from-python")
    assert await connection.recv_text() == "from-server"
    await connection.close()
    assert await connection.recv_text() is None

asyncio.run(asyncio.wait_for(exercise(), timeout=5))
sys.modules.pop('litellm.rust_bridge._native')
sys.modules.pop('litellm.rust_bridge')
sys.modules.pop('litellm')
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
