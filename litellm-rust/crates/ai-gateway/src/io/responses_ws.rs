use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{Sink, SinkExt, Stream, StreamExt};
use litellm_core::providers::openai::responses::transformation::OPENAI_RESPONSES_WS_CONFIG;
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::responses::websocket::ResponsesWebSocketProviderConfig;
use litellm_core::{CoreError, CoreResult};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
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
pub async fn responses_ws<In, Out>(
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
    splice(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
