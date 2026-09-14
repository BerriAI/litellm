use std::collections::HashMap;
use std::io;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use rustls::{ClientConfig, RootCertStore};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::error::TlsError;
use tokio_tungstenite::tungstenite::handshake::client::Response;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::{
    Connector, MaybeTlsStream, WebSocketStream, connect_async_tls_with_config,
};

use crate::Error;
use crate::constants::{OPENAI_RESPONSES_DEFAULT_API_BASE, OPENAI_RESPONSES_PATH};
use crate::responses::types::{ResponsesWsEvent, ResponsesWsEventType, ResponsesWsTransformResult};

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn supports_native_websocket(&self) -> bool {
        false
    }

    fn model_in_websocket_url(&self) -> bool {
        true
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String {
        complete_websocket_url(api_base, model, self.model_in_websocket_url())
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;
}

pub fn complete_websocket_url(
    api_base: Option<&str>,
    model: &str,
    model_in_websocket_url: bool,
) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let (base_without_query, query) = base
        .split_once('?')
        .map_or((base, None), |(value, query)| (value, Some(query)));
    let response_url = format!(
        "{}{}",
        base_without_query.trim_end_matches('/'),
        OPENAI_RESPONSES_PATH
    );
    let scheme_flipped = if let Some(rest) = response_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = response_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        response_url
    };
    let url = query.map_or(scheme_flipped.clone(), |value| {
        format!("{scheme_flipped}?{value}")
    });
    if !model_in_websocket_url
        || query.is_some_and(|value| {
            value
                .split('&')
                .any(|part| part.split('=').next() == Some("model"))
        })
    {
        return url;
    }
    format!(
        "{url}{}model={}",
        if query.is_some() { "&" } else { "?" },
        percent_encode(model)
    )
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                format!("{}", byte as char)
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

pub fn enforce_model(event: &ResponsesWsEvent, model: &str) -> ResponsesWsEvent {
    if !event.is_response_create() {
        return event.clone();
    }
    let mut enforced = event.clone();
    let has_flat_model = enforced.data.contains_key("model");
    if let Some(response) = enforced
        .data
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        response.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
        if has_flat_model {
            enforced.data.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }
    } else {
        enforced.data.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
    }
    enforced
}

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

pub type ResponsesUpstreamWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

static TLS_CONFIG: OnceLock<Arc<ClientConfig>> = OnceLock::new();

fn build_tls_config() -> Result<ClientConfig, Box<tokio_tungstenite::tungstenite::Error>> {
    let native = rustls_native_certs::load_native_certs();
    let mut store = RootCertStore::empty();
    let (added, _ignored) = store.add_parsable_certificates(native.certs);
    if added == 0 {
        return Err(Box::new(tokio_tungstenite::tungstenite::Error::Io(
            io::Error::other(format!(
                "no usable native root certificates: {:?}",
                native.errors
            )),
        )));
    }
    ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map(|builder| builder.with_root_certificates(store).with_no_client_auth())
        .map_err(|error| {
            Box::new(tokio_tungstenite::tungstenite::Error::Tls(
                TlsError::Rustls(error),
            ))
        })
}

fn tls_config() -> Result<Arc<ClientConfig>, Box<tokio_tungstenite::tungstenite::Error>> {
    if let Some(config) = TLS_CONFIG.get() {
        return Ok(Arc::clone(config));
    }
    let built = Arc::new(build_tls_config()?);
    Ok(Arc::clone(TLS_CONFIG.get_or_init(|| built)))
}

pub async fn connect_upstream<R>(
    request: R,
) -> Result<(ResponsesUpstreamWs, Response), Box<tokio_tungstenite::tungstenite::Error>>
where
    R: IntoClientRequest + Unpin,
{
    let request = request.into_client_request().map_err(Box::new)?;
    let connector = match request.uri().scheme_str() {
        Some("wss") => Some(Connector::Rustls(tls_config()?)),
        _ => None,
    };
    connect_async_tls_with_config(request, None, false, connector)
        .await
        .map_err(Box::new)
}

#[derive(Clone)]
pub struct ResponsesWebSocketConnection {
    socket: Arc<Mutex<Option<ResponsesUpstreamWs>>>,
}

impl ResponsesWebSocketConnection {
    pub async fn connect_url(
        url: &str,
        headers: &HashMap<String, String>,
        timeout: Option<Duration>,
    ) -> Result<Self, Error> {
        let mut request = url
            .into_client_request()
            .map_err(|error| Error::Network(error.to_string()))?;
        for (name, value) in headers {
            let header_name = name
                .parse::<HeaderName>()
                .map_err(|error| Error::InvalidRequest(error.to_string()))?;
            let header_value = HeaderValue::from_str(value)
                .map_err(|error| Error::InvalidRequest(error.to_string()))?;
            request.headers_mut().insert(header_name, header_value);
        }
        let connect = connect_upstream(request);
        let result = match timeout {
            Some(timeout) => tokio::time::timeout(timeout, connect)
                .await
                .map_err(|_| Error::Network("Responses WebSocket connection timed out".into()))?,
            None => connect.await,
        };
        let (socket, _) = result.map_err(|error| match *error {
            tokio_tungstenite::tungstenite::Error::Http(response) => Error::Http {
                status: response.status().as_u16(),
                body: String::new(),
            },
            other => Error::Network(other.to_string()),
        })?;
        Ok(Self {
            socket: Arc::new(Mutex::new(Some(socket))),
        })
    }

    pub async fn send_text(&self, text: String) -> Result<(), Error> {
        let mut socket = self.socket.lock().await;
        let Some(socket) = socket.as_mut() else {
            return Err(Error::Network("Responses WebSocket is closed".into()));
        };
        socket
            .send(Message::Text(text))
            .await
            .map_err(|error| Error::Network(error.to_string()))
    }

    pub async fn recv_text(&self) -> Result<Option<String>, Error> {
        let mut socket = self.socket.lock().await;
        let Some(socket) = socket.as_mut() else {
            return Ok(None);
        };
        match socket.next().await {
            Some(Ok(Message::Text(text))) => Ok(Some(text)),
            Some(Ok(Message::Binary(bytes))) => String::from_utf8(bytes.to_vec())
                .map(Some)
                .map_err(|error| Error::InvalidResponse(error.to_string())),
            Some(Ok(Message::Close(_))) | None => Ok(None),
            Some(Ok(_)) => Ok(None),
            Some(Err(error)) => Err(Error::Network(error.to_string())),
        }
    }

    pub async fn close(&self) -> Result<(), Error> {
        let mut socket = self.socket.lock().await;
        if let Some(socket) = socket.as_mut() {
            socket
                .close(None)
                .await
                .map_err(|error| Error::Network(error.to_string()))?;
        }
        *socket = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid event")
    }

    #[test]
    fn url_construction_matches_python_defaults_and_query_behavior() {
        assert_eq!(
            complete_websocket_url(None, "gpt-5", true),
            "wss://api.openai.com/v1/responses?model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("http://localhost:8080/"), "gpt 5", true),
            "ws://localhost:8080/responses?model=gpt%205"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test/v1?foo=bar"), "gpt-5", true),
            "wss://example.test/v1/responses?foo=bar&model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test?model=existing"), "gpt-5", true),
            "wss://example.test/responses?model=existing"
        );
    }

    #[test]
    fn enforce_model_overrides_flat_and_nested_values() {
        let flat = enforce_model(
            &event(serde_json::json!({"type":"response.create","model":"wrong"})),
            "gpt-5",
        );
        assert_eq!(flat.model(), Some("gpt-5"));
        let nested = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "model":"wrong",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert_eq!(nested.model(), Some("gpt-5"));
        assert_eq!(
            nested
                .data
                .get("response")
                .and_then(|value| value.get("model")),
            Some(&serde_json::json!("gpt-5"))
        );
        let nested_without_flat = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert!(!nested_without_flat.data.contains_key("model"));
    }
}
