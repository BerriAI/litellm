mod host;

use litellm_core::responses::websocket::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use serde_json::Value;

use crate::{
    errors::{RustBridgeDeclined, route_error_to_pyerr},
    marshal::{marshal_headers, optional_timeout},
};

fn run_public(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    use super::inference::InferenceHost;
    use litellm_callbacks_legacy_python::LoggingOperation;
    let host = InferenceHost::new(
        request.clone().unbind(),
        "litellm.rust_bridge.responses.route_host",
    );
    if let Some(reason) = py
        .import("litellm.rust_bridge.responses.route_host")?
        .getattr("decline_reason")?
        .call1((&request,))?
        .extract::<Option<String>>()?
    {
        return Err(RustBridgeDeclined::new_err(reason));
    }
    let model = host
        .argument(py, &kwargs, "model")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("model is required"))?
        .extract::<String>()?;
    let provider = host
        .argument(py, &kwargs, "custom_llm_provider")?
        .map(|value| value.extract::<String>())
        .transpose()?;
    if provider
        .as_deref()
        .is_some_and(|provider| provider != "openai")
        || model
            .strip_prefix("openai/")
            .unwrap_or(&model)
            .contains('/')
    {
        return Err(RustBridgeDeclined::new_err(
            "native HTTP responses provider",
        ));
    }
    if host
        .argument(py, &kwargs, "stream")?
        .map(|value| litellm_host_python::from_py::<Value>(&value))
        .transpose()?
        .is_some_and(|value| value == Value::Bool(true))
    {
        return Err(RustBridgeDeclined::new_err(
            "native Python responses streaming",
        ));
    }
    let cache_call_type = if asynchronous {
        "aresponses"
    } else {
        "responses"
    };
    crate::cache::admit_native(py, &kwargs, cache_call_type)?;
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        LoggingOperation::Responses,
        &request,
        &args,
        &kwargs,
        asynchronous,
    )?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = litellm_core::responses::ResponsesRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            let (cache, cache_options) =
                crate::cache::configured_native(py, arguments, cache_call_type)?;
            let route = match cache {
                Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
                    cache,
                    litellm_cache_response::CacheScope::Shared,
                )),
                None => route,
            };
            Ok(route.machine(request, cache_options.policy))
        },
        host::ResponsesPythonHost(host),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn responses(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_public(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn aresponses(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_public(py, request, args, kwargs, true)
}

#[pyclass]
pub(crate) struct ResponsesWebSocketConnection {
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
        #[pyo3(from_py_with = litellm_host_python::from_py_argument)] headers: Option<Value>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = marshal_headers(headers)?;
        let timeout = optional_timeout(timeout_seconds);
        crate::execution::run_async_value(py, async move {
            let inner = RustResponsesWebSocketConnection::connect_url(&url, &headers, timeout)
                .await
                .map_err(route_error_to_pyerr)?;
            Ok(ResponsesWebSocketConnection { inner })
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        crate::execution::run_async_value(py, async move {
            inner.send_text(text).await.map_err(route_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        crate::execution::run_async_value(py, async move {
            inner.recv_text().await.map_err(route_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        crate::execution::run_async_value(py, async move {
            inner.close().await.map_err(route_error_to_pyerr)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, time::Duration};

    use futures_util::{SinkExt, StreamExt};
    use pyo3::{prelude::*, types::PyDict};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "the test server shares the routes' runtime"
    )]
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
            let locals = PyDict::new(py);
            locals
                .set_item("native", crate::native_module(py))
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
