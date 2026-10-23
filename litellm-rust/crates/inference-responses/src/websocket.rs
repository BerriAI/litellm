use std::{collections::HashMap, sync::Arc, time::Duration};

use futures_util::{SinkExt, StreamExt};
use litellm_http::{
    Client,
    websocket::{self, UpstreamWebSocket},
};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::{
    self, Message,
    http::{HeaderMap, HeaderName, HeaderValue},
};

use super::Error;

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
        client: &Client,
        url: &str,
        headers: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        litellm_inference::diagnostic::operation("litellm.websocket.connect_url", async {
            let headers = headers
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.parse::<HeaderName>()
                            .map_err(|error| Error::InvalidRequest(error.to_string().into()))?,
                        HeaderValue::from_str(value)
                            .map_err(|error| Error::InvalidRequest(error.to_string().into()))?,
                    ))
                })
                .collect::<Result<HeaderMap, Error>>()?;
            let socket = websocket::connect(client, url, headers, timeout).await?;
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
        litellm_inference::diagnostic::operation("litellm.websocket.send_text", async {
            let mut socket = self.socket.lock().await;
            let Some(socket) = socket.as_mut() else {
                return Err(litellm_http::Error::from(tungstenite::Error::AlreadyClosed).into());
            };
            socket
                .send(Message::Text(text))
                .await
                .map_err(|error| litellm_http::Error::from(error).into())
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
        litellm_inference::diagnostic::operation("litellm.websocket.recv_text", async {
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
                Some(Err(error)) => Err(litellm_http::Error::from(error).into()),
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
        litellm_inference::diagnostic::operation("litellm.websocket.close", async {
            let mut socket = self.socket.lock().await;
            if let Some(socket) = socket.as_mut() {
                socket
                    .close(None)
                    .await
                    .map_err(litellm_http::Error::from)?;
            }
            *socket = None;
            Ok(())
        })
        .await
    }
}
