use std::{collections::HashMap, sync::Arc, time::Duration};

use futures_util::{SinkExt, StreamExt};
use litellm_http::websocket::{UpstreamWebSocket, connect_upstream};
use litellm_llms_types::responses::streaming_websocket::ResponsesWsEventType;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::{
    Message,
    client::IntoClientRequest,
    http::{HeaderName, HeaderValue},
};

use super::Error;

pub fn is_terminal_event(event_type: &ResponsesWsEventType) -> bool {
    matches!(
        event_type,
        ResponsesWsEventType::ResponseCreated
            | ResponsesWsEventType::ResponseCompleted
            | ResponsesWsEventType::ResponseFailed
            | ResponsesWsEventType::ResponseIncomplete
            | ResponsesWsEventType::Error
    )
}

#[derive(Clone)]
pub struct ResponsesWebSocketConnection {
    socket: Arc<Mutex<Option<UpstreamWebSocket>>>,
}

impl ResponsesWebSocketConnection {
    #[tracing::instrument(
        name = "litellm.websocket.connect_url",
        level = "debug",
        skip_all,
        fields(outcome)
    )]
    pub async fn connect_url(
        url: &str,
        headers: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        crate::diagnostic::operation("litellm.websocket.connect_url", async {
            let mut request = url.into_client_request().map_err(|error| {
                Error::Transport(litellm_http::transport::Error::Network(error.to_string()))
            })?;
            for (name, value) in headers {
                let header_name = name
                    .parse::<HeaderName>()
                    .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
                let header_value = HeaderValue::from_str(value)
                    .map_err(|error| Error::InvalidRequest(error.to_string().into()))?;
                request.headers_mut().insert(header_name, header_value);
            }
            let connect = connect_upstream(request);
            let result = match timeout {
                Some(timeout) => tokio::time::timeout(timeout, connect).await.map_err(|_| {
                    Error::Transport(litellm_http::transport::Error::Network(
                        "Responses WebSocket connection timed out".into(),
                    ))
                })?,
                None => connect.await,
            };
            let (socket, _) = result.map_err(|error| match *error {
                tokio_tungstenite::tungstenite::Error::Http(response) => {
                    Error::Transport(litellm_http::transport::Error::Http {
                        status: response.status().as_u16(),
                        body: String::new(),
                    })
                }
                other => {
                    Error::Transport(litellm_http::transport::Error::Network(other.to_string()))
                }
            })?;
            Ok(Self {
                socket: Arc::new(Mutex::new(Some(socket))),
            })
        })
        .await
    }

    #[tracing::instrument(
        name = "litellm.websocket.send_text",
        level = "debug",
        skip_all,
        fields(outcome)
    )]
    pub async fn send_text(&self, text: String) -> Result<(), Error> {
        crate::diagnostic::operation("litellm.websocket.send_text", async {
            let mut socket = self.socket.lock().await;
            let Some(socket) = socket.as_mut() else {
                return Err(Error::Transport(litellm_http::transport::Error::Network(
                    "Responses WebSocket is closed".into(),
                )));
            };
            socket.send(Message::Text(text)).await.map_err(|error| {
                Error::Transport(litellm_http::transport::Error::Network(error.to_string()))
            })
        })
        .await
    }

    #[tracing::instrument(
        name = "litellm.websocket.recv_text",
        level = "debug",
        skip_all,
        fields(outcome)
    )]
    pub async fn recv_text(&self) -> Result<Option<String>, Error> {
        crate::diagnostic::operation("litellm.websocket.recv_text", async {
            let mut socket = self.socket.lock().await;
            let Some(socket) = socket.as_mut() else {
                return Ok(None);
            };
            match socket.next().await {
                Some(Ok(Message::Text(text))) => Ok(Some(text)),
                Some(Ok(Message::Binary(bytes))) => String::from_utf8(bytes.to_vec())
                    .map(Some)
                    .map_err(|error| Error::InvalidResponse(error.to_string().into())),
                Some(Ok(Message::Close(_))) | None => Ok(None),
                Some(Ok(_)) => Ok(None),
                Some(Err(error)) => Err(Error::Transport(litellm_http::transport::Error::Network(
                    error.to_string(),
                ))),
            }
        })
        .await
    }

    #[tracing::instrument(
        name = "litellm.websocket.close",
        level = "debug",
        skip_all,
        fields(outcome)
    )]
    pub async fn close(&self) -> Result<(), Error> {
        crate::diagnostic::operation("litellm.websocket.close", async {
            let mut socket = self.socket.lock().await;
            if let Some(socket) = socket.as_mut() {
                socket.close(None).await.map_err(|error| {
                    Error::Transport(litellm_http::transport::Error::Network(error.to_string()))
                })?;
            }
            *socket = None;
            Ok(())
        })
        .await
    }
}
