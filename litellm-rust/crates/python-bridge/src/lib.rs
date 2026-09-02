mod constants;
mod diagnostics;
mod errors;
mod execution;
pub mod function_trace;
mod marshal;
mod routes;

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{marshal_headers, optional_timeout};

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustResponsesWebSocketConnection,
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (url, headers=None, timeout_seconds=None))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        url: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)] headers: Option<Value>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = marshal_headers(headers)?;
        let timeout = optional_timeout(timeout_seconds);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustResponsesWebSocketConnection::connect_url(&url, &headers, timeout)
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

#[pymodule(gil_used = false)]
mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        module.add_class::<super::ResponsesWebSocketConnection>()?;
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
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions_decline",
                "chat_completions",
                "achat_completions",
                "ResponsesWebSocketConnection",
                "gil_stats",
            ];

            let public_names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with("__"))
                .collect();
            assert_eq!(public_names, expected);
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
    connection = await native.ResponsesWebSocketConnection.connect(url)
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
