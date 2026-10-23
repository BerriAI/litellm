use http::{
    HeaderMap, HeaderValue, StatusCode,
    header::{
        CONNECTION, SEC_WEBSOCKET_ACCEPT, SEC_WEBSOCKET_KEY, SEC_WEBSOCKET_PROTOCOL,
        SEC_WEBSOCKET_VERSION, UPGRADE,
    },
};
use reqwest::Url;
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        handshake::{client::generate_key, derive_accept_key},
        protocol::Role,
    },
};

use crate::{client::Client, transport::Error};

pub type UpstreamWebSocket = WebSocketStream<reqwest::Upgraded>;

#[tracing::instrument(name = "litellm.websocket.handshake", level = "debug", skip_all)]
pub async fn connect(
    client: &Client,
    url: &str,
    headers: HeaderMap,
) -> Result<UpstreamWebSocket, Error> {
    let offered_protocols = headers.get(SEC_WEBSOCKET_PROTOCOL).cloned();
    let key = generate_key();
    let response = client
        .get(http_url(url)?)
        .headers(handshake_headers(headers, &key)?)
        .send()
        .await
        .map_err(Error::from_reqwest_before_dispatch)?;
    let status = response.status();
    if status != StatusCode::SWITCHING_PROTOCOLS {
        return Err(Error::Http {
            status: status.as_u16(),
            body: response.text().await.unwrap_or_default(),
        });
    }
    let expected = derive_accept_key(key.as_bytes());
    if response
        .headers()
        .get(SEC_WEBSOCKET_ACCEPT)
        .map(HeaderValue::as_bytes)
        != Some(expected.as_bytes())
    {
        return Err(Error::Network(
            "WebSocket handshake returned a wrong Sec-WebSocket-Accept".into(),
        ));
    }
    validate_response_headers(response.headers(), offered_protocols.as_ref())?;
    let upgraded = response.upgrade().await?;
    Ok(WebSocketStream::from_raw_socket(upgraded, Role::Client, None).await)
}

fn validate_response_headers(
    headers: &HeaderMap,
    offered_protocols: Option<&HeaderValue>,
) -> Result<(), Error> {
    if !headers
        .get(UPGRADE)
        .is_some_and(|value| value.as_bytes().eq_ignore_ascii_case(b"websocket"))
    {
        return Err(Error::Network(
            "WebSocket handshake returned an invalid Upgrade header".into(),
        ));
    }
    if !headers.get_all(CONNECTION).iter().any(|value| {
        value.to_str().is_ok_and(|value| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
        })
    }) {
        return Err(Error::Network(
            "WebSocket handshake returned an invalid Connection header".into(),
        ));
    }
    let protocol_matches = match (offered_protocols, headers.get(SEC_WEBSOCKET_PROTOCOL)) {
        (None, None) => true,
        (Some(offered), Some(selected)) => offered
            .to_str()
            .ok()
            .zip(selected.to_str().ok())
            .is_some_and(|(offered, selected)| {
                offered
                    .split(',')
                    .any(|protocol| protocol.trim() == selected)
            }),
        _ => false,
    };
    if !protocol_matches {
        return Err(Error::Network(
            "WebSocket handshake returned an invalid subprotocol".into(),
        ));
    }
    Ok(())
}

fn http_url(url: &str) -> Result<Url, Error> {
    let mut url = Url::parse(url).map_err(not_dispatched)?;
    let scheme = match url.scheme() {
        "ws" | "http" => "http",
        "wss" | "https" => "https",
        other => {
            return Err(not_dispatched(format!(
                "unsupported WebSocket URL scheme {other:?}"
            )));
        }
    };
    url.set_scheme(scheme)
        .map_err(|()| not_dispatched("WebSocket URL cannot be sent over HTTP"))?;
    Ok(url)
}

fn handshake_headers(headers: HeaderMap, key: &str) -> Result<HeaderMap, Error> {
    let key = HeaderValue::from_str(key).map_err(not_dispatched)?;
    Ok([
        (CONNECTION, HeaderValue::from_static("Upgrade")),
        (UPGRADE, HeaderValue::from_static("websocket")),
        (SEC_WEBSOCKET_VERSION, HeaderValue::from_static("13")),
        (SEC_WEBSOCKET_KEY, key),
    ]
    .into_iter()
    .fold(headers, |mut headers, (name, value)| {
        headers.insert(name, value);
        headers
    }))
}

fn not_dispatched(error: impl std::fmt::Display) -> Error {
    Error::Connect(error.to_string())
}
