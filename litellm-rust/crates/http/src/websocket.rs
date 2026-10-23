use std::time::Duration;

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

use crate::{
    client::Client,
    error::{Error, Repr},
};

pub type UpstreamWebSocket = WebSocketStream<reqwest::Upgraded>;

#[tracing::instrument(name = "litellm.websocket.handshake", level = "debug", skip_all)]
pub async fn connect(
    client: &Client,
    url: &str,
    headers: HeaderMap,
    timeout: Option<Duration>,
) -> Result<UpstreamWebSocket, Error> {
    match timeout {
        Some(timeout) => tokio::time::timeout(timeout, handshake(client, url, headers))
            .await
            .map_err(|_| Repr::HandshakeTimeout)?,
        None => handshake(client, url, headers).await,
    }
}

async fn handshake(
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
        .await?;
    let status = response.status();
    if status != StatusCode::SWITCHING_PROTOCOLS {
        return Err(Error::upstream(
            status.as_u16(),
            response.text().await.unwrap_or_default(),
        ));
    }
    let expected = derive_accept_key(key.as_bytes());
    if response
        .headers()
        .get(SEC_WEBSOCKET_ACCEPT)
        .map(HeaderValue::as_bytes)
        != Some(expected.as_bytes())
    {
        return Err(Repr::HandshakeAccept.into());
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
        return Err(Repr::HandshakeUpgrade.into());
    }
    if !headers.get_all(CONNECTION).iter().any(|value| {
        value.to_str().is_ok_and(|value| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
        })
    }) {
        return Err(Repr::HandshakeConnection.into());
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
        return Err(Repr::HandshakeSubprotocol.into());
    }
    Ok(())
}

fn http_url(url: &str) -> Result<Url, Error> {
    let mut url = Url::parse(url).map_err(Repr::WebSocketUrl)?;
    let scheme = match url.scheme() {
        "ws" | "http" => "http",
        "wss" | "https" => "https",
        other => return Err(Repr::WebSocketScheme(other.to_owned()).into()),
    };
    url.set_scheme(scheme)
        .map_err(|()| Repr::WebSocketScheme(url.scheme().to_owned()))?;
    Ok(url)
}

fn handshake_headers(headers: HeaderMap, key: &str) -> Result<HeaderMap, Error> {
    let key = HeaderValue::from_str(key).map_err(|error| Repr::Request(error.into()))?;
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
