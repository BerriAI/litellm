mod auth;
mod cache;
mod constants;
mod diagnostics;
mod errors;
mod execution;
#[cfg(feature = "trace-parity")]
mod function_trace;
mod lifecycle;
mod marshal;
mod routes;

use pyo3::prelude::*;

#[pymodule(gil_used = true)]
mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        super::diagnostics::register(module)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use pyo3::types::PyDict;
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);

            let expected = [
                "RustBridgeUnavailable",
                "RustBridgeDeclined",
                "RustHostCallbackError",
                "RustUpstreamError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions",
                "achat_completions",
                "embedding",
                "aembedding",
                "image_edit",
                "aimage_edit",
                "image_generation",
                "aimage_generation",
                "moderation",
                "amoderation",
                "rerank",
                "arerank",
                "ResponsesWebSocketConnection",
                "responses",
                "aresponses",
                "speech",
                "aspeech",
                "count_input_tokens",
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
            let locals = PyDict::new(py);
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
    connection = await native.ResponsesWebSocketConnection.connect(url, custom_llm_provider="openai")
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
