//! The streaming Anthropic Messages route: `amessages_stream`.
//!
//! Returns a Python async iterator (`MessagesStream`) that yields complete
//! SSE frames as `bytes`. There is deliberately no sync twin: the Python
//! consumer of this path is async-only, and no `bridge_route!` macro because
//! the route's value is a stream, not a terminal `Serialize`d response.

use std::sync::Arc;

use futures_util::StreamExt;
use litellm_core::http_utils::SseFrameStream;
use litellm_core::messages::messages_stream_frames;
use litellm_core::messages::types::MessagesRequest;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, required_value};

/// One complete SSE frame, converted to `bytes` on the attached thread that
/// completes the awaitable (mirrors `Pythonized` in litellm-python-interop).
struct SseFrame(Vec<u8>);

impl<'py> IntoPyObject<'py> for SseFrame {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(PyBytes::new(py, &self.0).into_any())
    }
}

/// Async iterator over the upstream SSE frames.
///
/// Pull-based: each `__anext__` awaits exactly one frame, so backpressure is
/// structural (nothing buffers beyond the current frame). Dropping the object
/// (GC, `break` out of `async for`) drops the upstream response body, which
/// aborts the provider request. Cancelling a task awaiting `__anext__` only
/// drops the frame fetch, so a later `__anext__` resumes without data loss.
#[pyclass]
struct MessagesStream {
    frames: Arc<Mutex<Option<SseFrameStream>>>,
}

#[pymethods]
impl MessagesStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let frames = Arc::clone(&self.frames);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = frames.lock().await;
            let frame = match guard.as_mut() {
                Some(stream) => stream.next().await,
                None => None,
            };
            match frame {
                Some(Ok(frame)) => Ok(SseFrame(frame)),
                Some(Err(error)) => {
                    // The upstream stream is dead after a mid-stream failure.
                    *guard = None;
                    Err(core_error_to_pyerr(error))
                }
                None => {
                    *guard = None;
                    Err(PyStopAsyncIteration::new_err(()))
                }
            }
        })
    }
}

#[pyfunction]
#[pyo3(signature = (model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None, trace=false))]
#[allow(clippy::too_many_arguments)]
fn amessages_stream<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] body: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    timeout_seconds: Option<f64>,
    trace: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // `trace_call` wraps a completed response; a stream has no terminal value
    // to attach the trace to. Reject rather than silently drop the request.
    if trace {
        return Err(PyValueError::new_err(
            "trace is not supported on the streaming route",
        ));
    }
    let body = required_value("body", body, Value::is_object, "dict")?;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout_seconds,
    })?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        let frames = messages_stream_frames(MessagesRequest {
            model: &model,
            body,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
        })
        .await
        .map_err(core_error_to_pyerr)?;
        Ok(MessagesStream {
            frames: Arc::new(Mutex::new(Some(frames))),
        })
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::routes::definition::add_function(
        module,
        pyo3::wrap_pyfunction!(amessages_stream, module)?,
    )?;
    module.add_class::<MessagesStream>()
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::time::Duration;

    use pyo3::types::{PyDict, PyModule};
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    use super::*;

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        use tokio::io::AsyncReadExt;

        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            let read = socket.read(&mut buffer).await.expect("reads request");
            if read == 0 {
                break request.len();
            }
            request.extend_from_slice(&buffer[..read]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        while request.len().saturating_sub(header_end) < content_length {
            let read = socket.read(&mut buffer).await.expect("reads body");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8(request).expect("request is utf8")
    }

    /// An SSE upstream that sends `body` in chunks separated by short delays
    /// so frames span network chunk boundaries.
    async fn streaming_upstream(
        listener: TcpListener,
        status_line: &'static str,
        body: &'static str,
    ) {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let request = read_http_request(&mut socket).await;
        assert!(
            request.contains("\"stream\":true"),
            "upstream body must force stream:true: {request}"
        );
        let response = format!(
            "{status_line}\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response head");
        let bytes = body.as_bytes();
        let chunks: Vec<&[u8]> = match bytes.len() {
            0..=30 => vec![bytes],
            31..=70 => vec![&bytes[..30], &bytes[30..]],
            _ => vec![&bytes[..30], &bytes[30..70], &bytes[70..]],
        };
        for chunk in chunks {
            tokio::time::sleep(Duration::from_millis(10)).await;
            socket.write_all(chunk).await.expect("writes body chunk");
        }
    }

    fn register_stream_module(py: Python<'_>) -> Bound<'_, PyModule> {
        let module = PyModule::new(py, "messages_stream").expect("module should be created");
        register(&module).expect("stream route should register");
        module
    }

    fn run_async_code(py: Python<'_>, module: &Bound<'_, PyModule>, url: &str, code: &str) {
        let locals = PyDict::new(py);
        locals
            .set_item("runtime", module)
            .expect("module should enter Python locals");
        locals
            .set_item("url", url)
            .expect("URL should enter Python locals");
        let source = CString::new(code).expect("Python source should not contain null bytes");
        py.run(&source, Some(&locals), Some(&locals))
            .expect("Python stream exercise should pass");
    }

    #[test]
    fn stream_route_yields_complete_frames_in_order() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        let body = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: content_block_delta\ndata: {\"delta\":\"hi\"}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let server = runtime.spawn(streaming_upstream(listener, "HTTP/1.1 200 OK", body));

        Python::attach(|py| {
            let module = register_stream_module(py);
            run_async_code(
                py,
                &module,
                &format!("http://{address}"),
                r#"
import asyncio

async def exercise():
    stream = await runtime.amessages_stream(
        model="claude-sonnet-4-5",
        body={"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []},
        api_key="sk-ant",
        api_base=url,
        custom_llm_provider="anthropic",
    )
    frames = [chunk async for chunk in stream]
    assert frames == [
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"delta":"hi"}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ], frames

asyncio.run(exercise())
"#,
            );
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }

    #[test]
    fn stream_route_surfaces_upstream_errors_before_the_stream() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        let server = runtime.spawn(streaming_upstream(
            listener,
            "HTTP/1.1 429 Too Many Requests",
            "{\"error\":\"rate limited\"}",
        ));

        Python::attach(|py| {
            let module = register_stream_module(py);
            run_async_code(
                py,
                &module,
                &format!("http://{address}"),
                r#"
import asyncio

async def exercise():
    try:
        await runtime.amessages_stream(
            model="claude-sonnet-4-5",
            body={"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []},
            api_key="sk-ant",
            api_base=url,
            custom_llm_provider="anthropic",
        )
    except RuntimeError as error:
        assert "429" in str(error), str(error)
    else:
        raise AssertionError("upstream error should surface from the awaitable")

asyncio.run(exercise())
"#,
            );
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }

    #[test]
    fn stream_route_ends_iteration_after_a_mid_stream_failure() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        // Advertise more bytes than are sent, then close: reqwest reports an
        // incomplete body mid-stream after the buffered frames are delivered.
        let server = runtime.spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let _request = read_http_request(&mut socket).await;
            let body = "event: message_start\ndata: {}\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len() + 40,
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            socket.shutdown().await.expect("closes socket");
        });

        Python::attach(|py| {
            let module = register_stream_module(py);
            run_async_code(
                py,
                &module,
                &format!("http://{address}"),
                r#"
import asyncio

async def exercise():
    stream = await runtime.amessages_stream(
        model="claude-sonnet-4-5",
        body={"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []},
        api_key="sk-ant",
        api_base=url,
        custom_llm_provider="anthropic",
    )
    frames = []
    failed = False
    try:
        async for chunk in stream:
            frames.append(chunk)
    except Exception:
        failed = True
    assert frames == [b'event: message_start\ndata: {}\n\n'], frames
    assert failed, "incomplete body should raise mid-stream"
    # After the failure the iterator is terminal.
    try:
        await stream.__anext__()
    except StopAsyncIteration:
        pass
    else:
        raise AssertionError("stream should be terminal after a failure")

asyncio.run(exercise())
"#,
            );
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }

    #[test]
    fn stream_route_rejects_trace() {
        Python::initialize();
        Python::attach(|py| {
            let module = register_stream_module(py);
            let error = amessages_stream(
                py,
                "claude-sonnet-4-5".to_string(),
                serde_json::json!({"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []}),
                None,
                None,
                None,
                None,
                None,
                true,
            )
            .expect_err("trace must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.to_string(),
                "ValueError: trace is not supported on the streaming route"
            );
            let _module = module;
        });
    }

    #[test]
    fn stream_route_supports_cancel_and_resume_without_frame_loss() {
        Python::initialize();
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let listener = runtime
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("listener should bind");
        let address = listener
            .local_addr()
            .expect("listener should have an address");
        // First frame arrives after a delay; a cancelled `__anext__` must not
        // consume it, and resuming must deliver it exactly once.
        let first_frame = "event: message_start\ndata: {}\n\n";
        let second_frame = "event: message_stop\ndata: {}\n\n";
        let total = first_frame.len() + second_frame.len();
        let server = runtime.spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let _request = read_http_request(&mut socket).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n"
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response head");
            tokio::time::sleep(Duration::from_millis(300)).await;
            socket
                .write_all(first_frame.as_bytes())
                .await
                .expect("writes first frame");
            tokio::time::sleep(Duration::from_millis(300)).await;
            socket
                .write_all(second_frame.as_bytes())
                .await
                .expect("writes second frame");
            socket.shutdown().await.expect("closes socket");
        });

        Python::attach(|py| {
            let module = register_stream_module(py);
            run_async_code(
                py,
                &module,
                &format!("http://{address}"),
                r#"
import asyncio

async def exercise():
    stream = await runtime.amessages_stream(
        model="claude-sonnet-4-5",
        body={"model": "claude-sonnet-4-5", "max_tokens": 8, "messages": []},
        api_key="sk-ant",
        api_base=url,
        custom_llm_provider="anthropic",
    )
    next_chunk = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.05)
    next_chunk.cancel()
    try:
        await next_chunk
    except asyncio.CancelledError:
        pass
    frames = [chunk async for chunk in stream]
    assert frames == [
        b'event: message_start\ndata: {}\n\n',
        b'event: message_stop\ndata: {}\n\n',
    ], frames

asyncio.run(asyncio.wait_for(exercise(), timeout=10))
"#,
            );
        });

        runtime
            .block_on(async { tokio::time::timeout(Duration::from_secs(5), server).await })
            .expect("server should finish")
            .expect("server task should not panic");
    }
}
