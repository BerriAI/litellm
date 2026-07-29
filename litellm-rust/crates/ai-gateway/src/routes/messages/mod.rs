//! `POST /v1/messages`, the Anthropic Messages HTTP surface.

mod service;

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use axum::Router;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};
use litellm_core::CoreError;
use serde_json::{Map, Value};

use crate::auth::RequireMasterKey;
use crate::constants::{
    BEDROCK_MESSAGES_PROVIDER, MESSAGES_HEADERS_NOT_FORWARDED, MESSAGES_ROUTE_PATH,
};
use crate::state::AppState;

/// This route's contribution to the app router.
pub fn router() -> Router<AppState> {
    Router::new().route(MESSAGES_ROUTE_PATH, post(handle))
}

async fn handle(
    _auth: RequireMasterKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, MessagesRouteError> {
    let extra_headers = forwarded_headers(&headers)?;
    match service::run(&state.router, body, extra_headers)
        .await
        .map_err(MessagesRouteError::from)?
    {
        service::MessagesResponse::Json(body) => Ok(Json(body).into_response()),
        service::MessagesResponse::Stream { provider, response } => {
            stream_response(provider, response)
        }
    }
}

fn stream_response(
    provider: String,
    upstream: reqwest::Response,
) -> Result<Response, MessagesRouteError> {
    let is_bedrock = provider == BEDROCK_MESSAGES_PROVIDER;
    let content_type = if is_bedrock {
        HeaderValue::from_static("text/event-stream")
    } else {
        upstream
            .headers()
            .get(CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("text/event-stream"))
    };
    let mut response = Response::builder()
        .status(
            StatusCode::from_u16(upstream.status().as_u16()).map_err(|error| {
                MessagesRouteError(CoreError::InvalidResponse(format!(
                    "invalid upstream response status: {error}"
                )))
            })?,
        )
        .header(CONTENT_TYPE, content_type);
    if let Some(value) = upstream.headers().get(CACHE_CONTROL) {
        response = response.header(CACHE_CONTROL, value);
    }
    let upstream_stream = upstream.bytes_stream().boxed();
    let body_stream = if is_bedrock {
        bedrock_sse_stream(upstream_stream)
    } else {
        upstream_stream
            .map(|result| result.map_err(|error| std::io::Error::other(error.to_string())))
            .boxed()
    };
    response
        .body(Body::from_stream(body_stream))
        .map_err(|error| {
            MessagesRouteError(CoreError::InvalidResponse(format!(
                "failed to build streaming response: {error}"
            )))
        })
}

struct EventStreamState {
    upstream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
    buffer: bytes::BytesMut,
    decoder: MessageFrameDecoder,
    terminated: bool,
}

fn bedrock_sse_stream(
    upstream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
) -> BoxStream<'static, Result<Bytes, std::io::Error>> {
    stream::unfold(
        EventStreamState {
            upstream,
            buffer: bytes::BytesMut::new(),
            decoder: MessageFrameDecoder::new(),
            terminated: false,
        },
        |mut state| async move {
            if state.terminated {
                return None;
            }
            loop {
                match state.decoder.decode_frame(&mut state.buffer) {
                    Ok(DecodedFrame::Complete(message)) => {
                        let bytes = message
                            .headers()
                            .iter()
                            .find(|header| header.name().as_str() == ":message-type")
                            .and_then(|header| header.value().as_string().ok())
                            .map_or_else(
                                || sse_data(message.payload()),
                                |message_type| {
                                    if message_type.as_str() == "exception"
                                        || message_type.as_str() == "error"
                                    {
                                        sse_error(message.payload())
                                    } else {
                                        sse_data(message.payload())
                                    }
                                },
                            );
                        return Some((Ok(Bytes::from(bytes)), state));
                    }
                    Ok(DecodedFrame::Incomplete) => {}
                    Err(error) => {
                        state.terminated = true;
                        return Some((
                            Ok(Bytes::from(sse_error(error.to_string().as_bytes()))),
                            state,
                        ));
                    }
                }
                match state.upstream.next().await {
                    Some(Ok(chunk)) => state.buffer.extend_from_slice(&chunk),
                    Some(Err(error)) => {
                        return Some((Err(std::io::Error::other(error.to_string())), state));
                    }
                    None if state.buffer.is_empty() => return None,
                    None => {
                        state.terminated = true;
                        return Some((
                            Ok(Bytes::from(sse_error(
                                b"incomplete Bedrock event stream frame",
                            ))),
                            state,
                        ));
                    }
                }
            }
        },
    )
    .boxed()
}

fn sse_data(payload: &[u8]) -> String {
    let value = serde_json::from_slice::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| value.get("bytes").and_then(serde_json::Value::as_str).map(str::to_string))
        .and_then(|encoded| {
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, encoded).ok()
        })
        .and_then(|payload| serde_json::from_slice::<serde_json::Value>(&payload).ok())
        .unwrap_or_else(|| serde_json::json!({"type": "error", "error": {"type": "invalid_request_error", "message": "invalid Bedrock event"}}));
    let event = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("message");
    format!("event: {event}\ndata: {value}\n\n")
}

fn sse_error(payload: &[u8]) -> String {
    let message = String::from_utf8_lossy(payload);
    format!(
        "event: error\ndata: {}\n\n",
        serde_json::json!({"type": "error", "error": {"type": "api_error", "message": message}})
    )
}

fn forwarded_headers(headers: &HeaderMap) -> Result<Option<Map<String, Value>>, CoreError> {
    let forwarded = headers
        .iter()
        .filter(|(name, _)| {
            !MESSAGES_HEADERS_NOT_FORWARDED
                .iter()
                .any(|excluded| name.as_str().eq_ignore_ascii_case(excluded))
        })
        .map(|(name, value)| {
            let value = value.to_str().map_err(|_| {
                CoreError::InvalidRequest(format!("invalid value for header {}", name.as_str()))
            })?;
            Ok((name.to_string(), Value::String(value.to_string())))
        })
        .collect::<Result<Map<_, _>, CoreError>>()?;
    Ok((!forwarded.is_empty()).then_some(forwarded))
}

#[derive(Debug)]
struct MessagesRouteError(CoreError);

impl From<CoreError> for MessagesRouteError {
    fn from(error: CoreError) -> Self {
        Self(error)
    }
}

impl IntoResponse for MessagesRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            CoreError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            CoreError::InvalidProvider(_) | CoreError::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no messages deployment is configured for this model".to_string(),
            ),
            CoreError::Auth(_) => (
                StatusCode::BAD_GATEWAY,
                "messages provider authentication failed".to_string(),
            ),
            CoreError::Http { .. }
            | CoreError::Network(_)
            | CoreError::InvalidResponse(_)
            | CoreError::InvalidType { .. }
            | CoreError::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "messages provider request failed".to_string(),
            ),
        };
        (
            status,
            Json(serde_json::json!({"error": {"message": message}})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
    use bytes::BytesMut;
    use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    use super::super::app;
    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;

    fn state(model: &str, api_base: String, master_key: Option<&str>) -> AppState {
        state_with_provider(model, model, api_base, master_key)
    }

    fn state_with_provider(
        model_alias: &str,
        provider_model: &str,
        api_base: String,
        master_key: Option<&str>,
    ) -> AppState {
        AppState {
            router: Arc::new(ModelRouter::new(vec![Deployment {
                model_name: model_alias.to_string(),
                litellm_params: LiteLLMParams {
                    model: format!("anthropic/{provider_model}"),
                    api_key: Some("upstream-key".to_string()),
                    api_base: Some(api_base),
                },
            }])),
            master_key: master_key.map(Arc::from),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        }
    }

    fn bedrock_state(
        model_alias: &str,
        provider_model: &str,
        api_base: String,
        api_key: Option<&str>,
        master_key: Option<&str>,
    ) -> AppState {
        AppState {
            router: Arc::new(ModelRouter::new(vec![Deployment {
                model_name: model_alias.to_string(),
                litellm_params: LiteLLMParams {
                    model: format!("bedrock/{provider_model}"),
                    api_key: api_key.map(str::to_string),
                    api_base: Some(api_base),
                },
            }])),
            master_key: master_key.map(Arc::from),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
        }
    }

    fn event_frame(message_type: &str, payload: serde_json::Value) -> Vec<u8> {
        let mut frame = BytesMut::new();
        let message = Message::new(
            serde_json::to_vec(&serde_json::json!({
                "bytes": base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    serde_json::to_vec(&payload).expect("payload"),
                )
            }))
            .expect("wrapper"),
        )
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String(message_type.to_string().into()),
        ));
        write_message_to(&message, &mut frame).expect("event frame");
        frame.to_vec()
    }

    async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = socket.read(&mut buffer).await.expect("reads request");
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("header end")
            + 4;
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
            request.extend_from_slice(&buffer[..read]);
        }
        String::from_utf8(request).expect("request utf8")
    }

    async fn bedrock_streaming_upstream(
        listener: TcpListener,
        frames: Vec<Vec<u8>>,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let address = listener.local_addr().expect("listener has address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let request = read_request(&mut socket).await;
            let body = frames.into_iter().flatten().collect::<Vec<_>>();
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/vnd.amazon.eventstream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket
                .write_all(head.as_bytes())
                .await
                .expect("writes head");
            let split = body.len() / 2;
            socket
                .write_all(&body[..split])
                .await
                .expect("writes first frame part");
            tokio::time::sleep(Duration::from_millis(10)).await;
            socket
                .write_all(&body[split..])
                .await
                .expect("writes second frame part");
            request
        });
        (format!("http://{address}"), server)
    }

    static AWS_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    async fn upstream(listener: TcpListener) -> (String, tokio::task::JoinHandle<String>) {
        let address = listener.local_addr().expect("listener has address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.expect("reads request");
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(request).expect("request is utf8");
            let content_length = request
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            let header_end = request.find("\r\n\r\n").expect("request has headers") + 4;
            let mut full_request = request.into_bytes();
            while full_request.len().saturating_sub(header_end) < content_length {
                let read = socket.read(&mut buffer).await.expect("reads body");
                full_request.extend_from_slice(&buffer[..read]);
            }
            let request = String::from_utf8(full_request).expect("request is utf8");
            let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-test"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });
        (format!("http://{address}"), server)
    }

    async fn streaming_upstream(
        listener: TcpListener,
        status: u16,
        content_type: &'static str,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let address = listener.local_addr().expect("listener has address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.expect("reads request");
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request_text = String::from_utf8(request).expect("request is utf8");
            let content_length = request_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            let header_end = request_text.find("\r\n\r\n").expect("request has headers") + 4;
            let mut full_request = request_text.into_bytes();
            while full_request.len().saturating_sub(header_end) < content_length {
                let read = socket.read(&mut buffer).await.expect("reads body");
                full_request.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 {status} OK\r\ncontent-type: {content_type}\r\ncache-control: no-cache\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            String::from_utf8(full_request).expect("request is utf8")
        });
        (format!("http://{address}"), server)
    }

    #[tokio::test]
    async fn route_constructs_anthropic_upstream_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let app = app(state("claude-test", api_base, Some("master-key")));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .header("x-api-key", "request-upstream-key")
                    .header("anthropic-beta", "beta-feature")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
                            "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("json")["id"],
            "msg_1"
        );
        let upstream_request = server.await.expect("upstream task completes");
        let (head, body) = upstream_request
            .split_once("\r\n\r\n")
            .expect("upstream request has body");
        let head = head.to_ascii_lowercase();
        assert!(head.contains("x-api-key: request-upstream-key"));
        assert!(head.contains("anthropic-beta: beta-feature"));
        assert!(!head.contains("authorization: bearer master-key"));
        let body: serde_json::Value = serde_json::from_str(body).expect("upstream body is json");
        assert_eq!(body["model"], "claude-test");
        assert_eq!(body["messages"][0]["content"], "hello");
    }

    #[tokio::test]
    async fn route_substitutes_model_alias_with_provider_model_upstream() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let app = app(state_with_provider(
            "production",
            "claude-sonnet-4-5",
            api_base,
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "production",
                            "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let upstream_request = server.await.expect("upstream task completes");
        let (_, upstream_body) = upstream_request
            .split_once("\r\n\r\n")
            .expect("upstream request has body");
        let upstream_body: serde_json::Value =
            serde_json::from_str(upstream_body).expect("upstream body is json");
        assert_eq!(upstream_body["model"], "claude-sonnet-4-5");
        assert_ne!(upstream_body["model"], "production");
    }

    #[tokio::test]
    async fn route_streams_anthropic_events_without_buffering_or_reordering() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let events = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let (api_base, server) =
            streaming_upstream(listener, 200, "text/event-stream", events).await;
        let app = app(state("claude-test", api_base, Some("master-key")));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
                            "max_tokens": 16,
                            "stream": true,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "text/event-stream"
        );
        assert_eq!(
            response
                .headers()
                .get(CACHE_CONTROL)
                .unwrap()
                .to_str()
                .unwrap(),
            "no-cache"
        );
        let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        assert_eq!(response_body, events.as_bytes());
        let upstream_request = server.await.expect("upstream task completes");
        let (_, upstream_body) = upstream_request
            .split_once("\r\n\r\n")
            .expect("upstream request has body");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(upstream_body)
                .expect("upstream body is json")["stream"],
            true
        );
    }

    #[tokio::test]
    async fn route_maps_streaming_upstream_errors_before_starting_response() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = streaming_upstream(
            listener,
            429,
            "application/json",
            r#"{"error":"rate limited"}"#,
        )
        .await;
        let app = app(state("claude-test", api_base, Some("master-key")));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
                            "max_tokens": 16,
                            "stream": true,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response_body).expect("error is json")["error"]
                ["message"],
            "messages provider request failed"
        );
        server.await.expect("upstream task completes");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn route_constructs_bedrock_sigv4_invoke_request() {
        let _lock = AWS_ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock");
        unsafe {
            std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-access-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret-key");
            std::env::set_var("AWS_REGION_NAME", "us-east-1");
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let model = "arn:aws:bedrock:us-east-1:123:model/foo";
        let app = app(bedrock_state(
            model,
            model,
            api_base,
            None,
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": model,
                            "stream": false,
                            "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let request = server.await.expect("server task completes");
        let (head, body) = request.split_once("\r\n\r\n").expect("request body");
        assert!(head.starts_with(
            "POST /model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123%3Amodel%2Ffoo/invoke "
        ));
        let head_lower = head.to_ascii_lowercase();
        assert!(head_lower.contains("authorization: aws4-hmac-sha256"));
        assert!(head_lower.contains("x-amz-date:"));
        let body: serde_json::Value = serde_json::from_str(body).expect("body json");
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert!(body.get("model").is_none());
        assert!(body.get("stream").is_none());
    }

    #[tokio::test]
    async fn route_constructs_bedrock_bearer_request_without_anthropic_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let app = app(bedrock_state(
            "claude-test",
            "claude-test",
            api_base,
            Some("bedrock-token"),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("x-api-key", "client-key")
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
                            "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let request = server.await.expect("server task completes");
        let (head, _) = request.split_once("\r\n\r\n").expect("request body");
        let head = head.to_ascii_lowercase();
        assert!(head.contains("authorization: bearer bedrock-token"));
        assert!(!head.contains("x-api-key:"));
        assert!(!head.contains("anthropic-version:"));
    }

    #[tokio::test]
    async fn route_decodes_bedrock_eventstream_split_frames_and_tool_use() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let frames = vec![
            event_frame("event", json!({"type": "message_start"})),
            event_frame(
                "event",
                json!({
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                    "index": 0
                }),
            ),
            event_frame(
                "event",
                json!({
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{\"city\":\"Paris\"}"}
                }),
            ),
            event_frame("event", json!({"type": "message_stop"})),
        ];
        let (api_base, server) = bedrock_streaming_upstream(listener, frames).await;
        let app = app(bedrock_state(
            "claude-test",
            "claude-test",
            api_base,
            Some("bedrock-token"),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
                            "stream": true,
                            "max_tokens": 16,
                            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                            "messages": [{"role": "user", "content": "weather"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");
        let body = String::from_utf8(body.to_vec()).expect("sse");
        assert!(body.contains("event: message_start"));
        assert!(body.contains("\"type\":\"tool_use\""));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("input_json_delta"));
        assert!(body.contains("event: message_stop"));
        let request = server.await.expect("server task completes");
        let (_, request_body) = request.split_once("\r\n\r\n").expect("request body");
        let request_body: serde_json::Value =
            serde_json::from_str(request_body).expect("request json");
        assert_eq!(request_body["tools"][0]["name"], "get_weather");
    }

    #[tokio::test]
    async fn route_maps_bedrock_errors_and_eventstream_exceptions() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let _ = read_request(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-length: 3\r\nconnection: close\r\n\r\nbad")
                .await
                .expect("write error");
        });
        let response = app(bedrock_state(
            "claude-test",
            "claude-test",
            format!("http://{address}"),
            Some("bedrock-token"),
            Some("master-key"),
        ))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("authorization", "Bearer master-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "claude-test", "max_tokens": 8, "messages": []}).to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("route responds");
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        server.await.expect("server task");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = bedrock_streaming_upstream(
            listener,
            vec![event_frame(
                "exception",
                json!({
                    "message": "model unavailable"
                }),
            )],
        )
        .await;
        let response = app(bedrock_state(
            "claude-test",
            "claude-test",
            api_base,
            Some("bedrock-token"),
            Some("master-key"),
        ))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("authorization", "Bearer master-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": "claude-test", "stream": true, "max_tokens": 8, "messages": []})
                        .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("route responds");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body");
        assert!(
            String::from_utf8(body.to_vec())
                .expect("sse")
                .contains("event: error")
        );
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn route_rejects_missing_master_key() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn route_rejects_invalid_master_key() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer wrong-key")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn route_rejects_malformed_json_without_panicking() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from("{not-json"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
