//! The upstream Responses WebSocket connection, consumed by hosts: the Python
//! bridge wraps it in a pyclass, and the gateway splices it to its own caller.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::http::header::HeaderName;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use crate::{CoreError, CoreResult};

pub type ResponsesUpstreamWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

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

    /// Receive the next data frame as text.
    ///
    /// `Ok(None)` means the connection is closed: a close frame arrived, the
    /// stream ended, or `close()` was already called. Control frames
    /// (ping/pong) are skipped, never surfaced as `None`; tungstenite queues
    /// the pong reply automatically when it yields a ping.
    pub async fn recv_text(&self) -> CoreResult<Option<String>> {
        let mut socket_guard = self.socket.lock().await;
        let Some(socket) = socket_guard.as_mut() else {
            return Ok(None);
        };
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => return Ok(Some(text)),
                Some(Ok(Message::Binary(bytes))) => {
                    return String::from_utf8(bytes.to_vec())
                        .map(Some)
                        .map_err(|error| CoreError::InvalidResponse(error.to_string()));
                }
                Some(Ok(Message::Close(_))) | None => return Ok(None),
                Some(Ok(_)) => {}
                Some(Err(error)) => return Err(CoreError::Network(error.to_string())),
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_async;

    /// A server that sends the scripted frames, then keeps the socket open
    /// (draining client frames) until the client disconnects.
    async fn connection_server(frames: Vec<Message>) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("local address");
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let mut socket = accept_async(stream).await.expect("websocket handshake");
            for frame in frames {
                socket.send(frame).await.expect("send frame");
            }
            while let Some(Ok(_)) = socket.next().await {}
        });
        (format!("ws://{address}"), task)
    }

    async fn connect(url: &str) -> ResponsesWebSocketConnection {
        ResponsesWebSocketConnection::connect_url(url, &HashMap::new(), None)
            .await
            .expect("connect")
    }

    #[tokio::test]
    async fn recv_text_returns_text_frames() {
        let (url, _server) = connection_server(vec![Message::Text("hello".to_string())]).await;
        let connection = connect(&url).await;
        assert_eq!(
            connection.recv_text().await.expect("recv"),
            Some("hello".to_string())
        );
    }

    #[tokio::test]
    async fn recv_text_decodes_binary_frames() {
        let (url, _server) = connection_server(vec![Message::Binary(b"binary".to_vec())]).await;
        let connection = connect(&url).await;
        assert_eq!(
            connection.recv_text().await.expect("recv"),
            Some("binary".to_string())
        );
    }

    #[tokio::test]
    async fn recv_text_skips_ping_frames() {
        let (url, _server) = connection_server(vec![
            Message::Ping(Vec::new()),
            Message::Text("after-ping".to_string()),
        ])
        .await;
        let connection = connect(&url).await;
        assert_eq!(
            connection.recv_text().await.expect("recv"),
            Some("after-ping".to_string())
        );
    }

    #[tokio::test]
    async fn recv_text_returns_none_on_close_frame() {
        let (url, _server) = connection_server(vec![Message::Close(None)]).await;
        let connection = connect(&url).await;
        assert_eq!(connection.recv_text().await.expect("recv"), None);
    }

    #[tokio::test]
    async fn recv_text_returns_none_after_local_close() {
        let (url, _server) = connection_server(Vec::new()).await;
        let connection = connect(&url).await;
        connection.close().await.expect("close");
        assert_eq!(connection.recv_text().await.expect("recv"), None);
    }
}
