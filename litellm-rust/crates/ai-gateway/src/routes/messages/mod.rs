//! `POST /v1/messages`, the Anthropic Messages HTTP surface.

mod service;

use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::header::{HeaderMap, HeaderValue, CACHE_CONTROL, CONTENT_TYPE};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use litellm_core::CoreError;
use serde_json::{Map, Value};

use crate::auth::RequireMasterKey;
use crate::constants::{MESSAGES_HEADERS_NOT_FORWARDED, MESSAGES_ROUTE_PATH};
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
        service::MessagesResponse::Stream(upstream) => stream_response(upstream),
    }
}

fn stream_response(upstream: reqwest::Response) -> Result<Response, MessagesRouteError> {
    let content_type = upstream
        .headers()
        .get(CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("text/event-stream"));
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
    response
        .body(Body::from_stream(upstream.bytes_stream()))
        .map_err(|error| {
            MessagesRouteError(CoreError::InvalidResponse(format!(
                "failed to build streaming response: {error}"
            )))
        })
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

    use axum::body::Body;
    use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
    use axum::http::Request;
    use axum::http::StatusCode;
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
            serde_json::from_slice::<serde_json::Value>(&response_body).expect("error is json")
                ["error"]["message"],
            "messages provider request failed"
        );
        server.await.expect("upstream task completes");
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
