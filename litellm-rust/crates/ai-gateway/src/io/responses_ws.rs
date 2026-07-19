use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use litellm_core::providers::openai::responses::transformation::OPENAI_RESPONSES_WS_CONFIG;
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::responses::websocket::ResponsesWebSocketProviderConfig;
use litellm_core::{CoreError, CoreResult};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{HeaderName, AUTHORIZATION};
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::constants::{
    DEFAULT_RESPONSES_WS_CONNECT_TIMEOUT_SECS, DEFAULT_RESPONSES_WS_IDLE_TIMEOUT_SECS,
};

const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const MISSING_KEY_MESSAGE: &str =
    "Missing OpenAI API Key - a Responses WebSocket call is being made but no key was passed via params or the OPENAI_API_KEY environment variable";

pub type ResponsesUpstreamWs = WebSocketStream<MaybeTlsStream<TcpStream>>;
type UpstreamTx = SplitSink<ResponsesUpstreamWs, Message>;
type UpstreamRx = SplitStream<ResponsesUpstreamWs>;

#[derive(Clone)]
pub struct ResponsesWebSocketConnection {
    socket: Arc<Mutex<Option<ResponsesUpstreamWs>>>,
}

impl ResponsesWebSocketConnection {
    pub async fn connect_url(
        url: &str,
        headers: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> CoreResult<Self> {
        let mut request = url
            .into_client_request()
            .map_err(|error| CoreError::Network(error.to_string()))?;
        for (name, value) in headers {
            let header_name = name
                .parse::<HeaderName>()
                .map_err(|error| CoreError::InvalidRequest(error.to_string()))?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|error| CoreError::InvalidRequest(error.to_string()))?;
            request.headers_mut().insert(header_name, header_value);
        }
        let connect = connect_async(request);
        let result = match timeout {
            Some(timeout) => tokio::time::timeout(timeout, connect).await.map_err(|_| {
                CoreError::Network("Responses WebSocket connection timed out".to_string())
            })?,
            None => connect.await,
        };
        let (socket, _) = result.map_err(|error| match error {
            tokio_tungstenite::tungstenite::Error::Http(response) => CoreError::Http {
                status: response.status().as_u16(),
                body: String::new(),
            },
            other => CoreError::Network(other.to_string()),
        })?;
        Ok(Self {
            socket: Arc::new(Mutex::new(Some(socket))),
        })
    }

    pub async fn send_text(&self, text: String) -> CoreResult<()> {
        let mut socket = self.socket.lock().await;
        let Some(socket) = socket.as_mut() else {
            return Err(CoreError::Network(
                "Responses WebSocket is closed".to_string(),
            ));
        };
        socket
            .send(Message::Text(text))
            .await
            .map_err(|error| CoreError::Network(error.to_string()))
    }

    pub async fn recv_text(&self) -> CoreResult<Option<String>> {
        let mut socket_guard = self.socket.lock().await;
        let Some(socket) = socket_guard.as_mut() else {
            return Ok(None);
        };
        match socket.next().await {
            Some(Ok(Message::Text(text))) => Ok(Some(text)),
            Some(Ok(Message::Binary(bytes))) => String::from_utf8(bytes.to_vec())
                .map(Some)
                .map_err(|error| CoreError::InvalidResponse(error.to_string())),
            Some(Ok(Message::Close(_))) | None => Ok(None),
            Some(Ok(_)) => Ok(None),
            Some(Err(error)) => Err(CoreError::Network(error.to_string())),
        }
    }

    pub async fn close(&self) -> CoreResult<()> {
        let mut socket = self.socket.lock().await;
        if let Some(socket) = socket.as_mut() {
            socket
                .close(None)
                .await
                .map_err(|error| CoreError::Network(error.to_string()))?;
        }
        *socket = None;
        Ok(())
    }
}

pub(crate) fn resolve_api_key(api_key: Option<&str>) -> CoreResult<String> {
    api_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(OPENAI_API_KEY_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .ok_or_else(|| CoreError::Auth(MISSING_KEY_MESSAGE.to_string()))
}

async fn dial_upstream(
    model: &str,
    api_key: &str,
    api_base: Option<&str>,
) -> CoreResult<ResponsesUpstreamWs> {
    let url = OPENAI_RESPONSES_WS_CONFIG.complete_websocket_url(api_base, model);
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|error| CoreError::Network(error.to_string()))?;
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|error| CoreError::Auth(error.to_string()))?,
    );
    let result = tokio::time::timeout(
        Duration::from_secs(DEFAULT_RESPONSES_WS_CONNECT_TIMEOUT_SECS),
        connect_async(request),
    )
    .await
    .map_err(|_| CoreError::Network("Responses WebSocket connection timed out".to_string()))?;
    result
        .map(|(socket, _)| socket)
        .map_err(|error| match error {
            tokio_tungstenite::tungstenite::Error::Http(response) => CoreError::Http {
                status: response.status().as_u16(),
                body: String::new(),
            },
            other => CoreError::Network(other.to_string()),
        })
}

pub struct ResponsesWebSocketStreaming;

impl ResponsesWebSocketStreaming {
    pub async fn bidirectional_forward<In, Out>(
        model: &str,
        upstream_tx: UpstreamTx,
        upstream_rx: UpstreamRx,
        idle_timeout: Option<Duration>,
        observe: impl FnMut(&ResponsesWsEvent) + Send,
        client_in: In,
        client_out: Out,
    ) -> CoreResult<()>
    where
        In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
        Out: Sink<ResponsesWsEvent> + Unpin + Send,
        Out::Error: std::fmt::Display,
    {
        splice(
            model,
            upstream_tx,
            upstream_rx,
            idle_timeout,
            observe,
            client_in,
            client_out,
        )
        .await
    }
}

pub(crate) async fn splice<In, Out>(
    model: &str,
    mut upstream_tx: UpstreamTx,
    mut upstream_rx: UpstreamRx,
    idle_timeout: Option<Duration>,
    mut observe: impl FnMut(&ResponsesWsEvent) + Send,
    mut client_in: In,
    mut client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let idle =
        idle_timeout.unwrap_or_else(|| Duration::from_secs(DEFAULT_RESPONSES_WS_IDLE_TIMEOUT_SECS));
    loop {
        tokio::select! {
            event = client_in.next() => {
                let Some(event) = event else { break };
                for outbound in OPENAI_RESPONSES_WS_CONFIG
                    .transform_ws_request(&event, model)?
                    .events
                {
                    let payload = serde_json::to_string(&outbound)
                        .map_err(|error| CoreError::InvalidResponse(error.to_string()))?;
                    upstream_tx.send(Message::Text(payload))
                        .await
                        .map_err(|error| CoreError::Network(error.to_string()))?;
                }
            }
            message = upstream_rx.next() => {
                let Some(message) = message else { break };
                match message.map_err(|error| CoreError::Network(error.to_string()))? {
                    Message::Text(text) => {
                        let event = serde_json::from_str::<ResponsesWsEvent>(&text)
                            .map_err(|error| CoreError::InvalidResponse(error.to_string()))?;
                        observe(&event);
                        for outbound in OPENAI_RESPONSES_WS_CONFIG
                            .transform_ws_response(&event, model)?
                            .events
                        {
                            client_out.send(outbound)
                                .await
                                .map_err(|error| CoreError::Network(error.to_string()))?;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            _ = tokio::time::sleep(idle) => break,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn async_responses_websocket<In, Out>(
    model: &str,
    api_key: Option<&str>,
    api_base: Option<&str>,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Option<Duration>,
    mut observe: impl FnMut(&ResponsesWsEvent) + Send,
    client_in: In,
    client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let key = resolve_api_key(api_key)?;
    let upstream = dial_upstream(model, &key, api_base).await?;
    let (mut upstream_tx, upstream_rx) = upstream.split();
    if let Some(first_frame) = first_frame {
        for outbound in OPENAI_RESPONSES_WS_CONFIG
            .transform_ws_request(&first_frame, model)?
            .events
        {
            let payload = serde_json::to_string(&outbound)
                .map_err(|error| CoreError::InvalidResponse(error.to_string()))?;
            upstream_tx
                .send(Message::Text(payload))
                .await
                .map_err(|error| CoreError::Network(error.to_string()))?;
        }
    }
    ResponsesWebSocketStreaming::bidirectional_forward(
        model,
        upstream_tx,
        upstream_rx,
        idle_timeout,
        &mut observe,
        client_in,
        client_out,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn responses_ws<In, Out>(
    model: &str,
    api_key: Option<&str>,
    api_base: Option<&str>,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Option<Duration>,
    observe: impl FnMut(&ResponsesWsEvent) + Send,
    client_in: In,
    client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    async_responses_websocket(
        model,
        api_key,
        api_base,
        first_frame,
        idle_timeout,
        observe,
        client_in,
        client_out,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_channel::mpsc;
    use futures_util::{SinkExt, StreamExt};
    use litellm_core::responses::types::ResponsesWsEventType;
    use serde_json::json;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    async fn websocket_base() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("local address");
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let mut socket = accept_async(stream).await.expect("websocket handshake");
            while let Some(Ok(Message::Text(text))) = socket.next().await {
                let request: serde_json::Value = serde_json::from_str(&text).expect("request json");
                let model = request
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| {
                        request
                            .get("response")
                            .and_then(serde_json::Value::as_object)
                            .and_then(|response| {
                                response.get("model").and_then(serde_json::Value::as_str)
                            })
                    })
                    .expect("enforced model");
                socket
                    .send(Message::Text(
                        json!({
                            "type": "response.created",
                            "response": {
                                "id": format!("resp-{model}"),
                                "model": model,
                                "extra": "preserved"
                            }
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("created event");
                socket
                    .send(Message::Text(
                        json!({
                            "type": "response.completed",
                            "response": {
                                "id": format!("resp-{model}"),
                                "model": model,
                                "usage": {
                                    "input_tokens": 1,
                                    "output_tokens": 2,
                                    "total_tokens": 3
                                }
                            }
                        })
                        .to_string(),
                    ))
                    .await
                    .expect("completed event");
            }
        });
        (format!("http://{address}"), task)
    }

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("event")
    }

    #[test]
    fn explicit_nonblank_key_wins() {
        assert_eq!(
            resolve_api_key(Some(" explicit ")).expect("key"),
            "explicit"
        );
    }

    #[test]
    fn blank_key_is_not_accepted_without_environment_key() {
        if std::env::var(OPENAI_API_KEY_ENV).is_err() {
            assert!(resolve_api_key(Some("  ")).is_err());
        }
    }

    #[tokio::test]
    async fn forwards_events_sequentially_and_enforces_model() {
        let (api_base, server) = websocket_base().await;
        let (client_tx, client_rx) = mpsc::unbounded();
        let (output_tx, mut output_rx) = mpsc::unbounded();
        let (observed_tx, observed_rx) = mpsc::unbounded();
        client_tx
            .unbounded_send(event(json!({
                "type": "response.create",
                "model": "wrong"
            })))
            .expect("first request");
        client_tx
            .unbounded_send(event(json!({
                "type": "response.create",
                "response": {"model": "also-wrong"}
            })))
            .expect("second request");

        let task = tokio::spawn(async move {
            responses_ws(
                "authorized-model",
                Some("test-key"),
                Some(&api_base),
                None,
                Some(Duration::from_secs(1)),
                move |event| {
                    observed_tx
                        .unbounded_send(event.clone())
                        .expect("observe event");
                },
                client_rx,
                output_tx,
            )
            .await
        });

        let first = output_rx.next().await.expect("first output");
        let second = output_rx.next().await.expect("second output");
        let third = output_rx.next().await.expect("third output");
        let fourth = output_rx.next().await.expect("fourth output");
        drop(client_tx);
        task.await.expect("splice task").expect("successful splice");
        server.await.expect("server task");

        assert_eq!(first.event_type, ResponsesWsEventType::ResponseCreated);
        assert_eq!(first.model(), Some("authorized-model"));
        assert_eq!(first.data["response"]["extra"], "preserved");
        assert_eq!(second.event_type, ResponsesWsEventType::ResponseCompleted);
        assert_eq!(third.event_type, ResponsesWsEventType::ResponseCreated);
        assert_eq!(fourth.event_type, ResponsesWsEventType::ResponseCompleted);
        let observed: Vec<_> = observed_rx.collect().await;
        assert_eq!(observed.len(), 4);
        assert!(observed
            .iter()
            .all(|event| event.event_type != ResponsesWsEventType::ResponseCreate));
    }

    #[tokio::test]
    async fn idle_timeout_ends_without_upstream_events() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let _socket = accept_async(stream).await.expect("handshake");
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        let (_client_tx, client_rx) = mpsc::unbounded::<ResponsesWsEvent>();
        let (output_tx, mut output_rx) = mpsc::unbounded();
        let result = responses_ws(
            "model",
            Some("key"),
            Some(&format!("http://{address}")),
            None,
            Some(Duration::from_millis(20)),
            |_| {},
            client_rx,
            output_tx,
        )
        .await;
        assert!(result.is_ok());
        assert!(output_rx.next().await.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn dial_http_status_is_preserved() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            stream
                .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                .await
                .expect("response");
        });
        let (_client_tx, client_rx) = mpsc::unbounded::<ResponsesWsEvent>();
        let (output_tx, _output_rx) = mpsc::unbounded();
        let error = responses_ws(
            "model",
            Some("key"),
            Some(&format!("http://{address}")),
            None,
            Some(Duration::from_millis(20)),
            |_| {},
            client_rx,
            output_tx,
        )
        .await
        .expect_err("status error");
        assert!(matches!(error, CoreError::Http { status: 401, .. }));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn dial_http_500_status_is_preserved() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            stream
                .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
                .await
                .expect("response");
        });
        let (_client_tx, client_rx) = mpsc::unbounded::<ResponsesWsEvent>();
        let (output_tx, _output_rx) = mpsc::unbounded();
        let error = responses_ws(
            "model",
            Some("key"),
            Some(&format!("http://{address}")),
            None,
            Some(Duration::from_millis(20)),
            |_| {},
            client_rx,
            output_tx,
        )
        .await
        .expect_err("status error");
        assert!(matches!(error, CoreError::Http { status: 500, .. }));
        server.await.expect("server task");
    }
}
