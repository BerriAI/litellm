//! `POST /v1/chat/completions`, the OpenAI-compatible chat completions HTTP surface.

mod service;

use axum::Router;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::http::header::HeaderMap;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::post;
use futures::stream::StreamExt;
use litellm_core::CoreError;
use serde_json::{Map, Value};

use crate::auth::key_auth::RequireValidKey;
use crate::constants::{CHAT_COMPLETIONS_HEADERS_NOT_FORWARDED, CHAT_COMPLETIONS_ROUTE_PATH};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route(CHAT_COMPLETIONS_ROUTE_PATH, post(handle))
}

async fn handle(
    auth: RequireValidKey,
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ChatCompletionsRouteError> {
    tracing::info!("Handler: Received request");
    let extra_headers = forwarded_headers(&headers)?;
    let result = service::run(
        &state,
        body,
        extra_headers,
        &auth.key_object,
        &auth.hashed_token,
    )
    .await
    .map_err(ChatCompletionsRouteError::from)?;

    match result {
        service::ChatCompletionsResult::Complete(bytes) => Ok((
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response()),
        service::ChatCompletionsResult::Streaming(stream) => {
            // Convert the stream of Result<Option<Value>> to SSE events
            let sse_stream = stream.filter_map(|chunk_result| async move {
                match chunk_result {
                    Ok(Some(value)) => {
                        // Format as SSE event
                        match serde_json::to_string(&value) {
                            Ok(data) => Some(Ok::<_, std::convert::Infallible>(
                                axum::response::sse::Event::default().data(data),
                            )),
                            Err(_) => None,
                        }
                    }
                    Ok(None) => {
                        // End of stream - send [DONE]
                        Some(Ok(axum::response::sse::Event::default().data("[DONE]")))
                    }
                    Err(_) => None, // Skip errors in stream
                }
            });
            Ok(Sse::new(sse_stream)
                .keep_alive(axum::response::sse::KeepAlive::default())
                .into_response())
        }
    }
}

fn forwarded_headers(headers: &HeaderMap) -> Result<Option<Map<String, Value>>, CoreError> {
    // Pre-allocate with estimated capacity (most requests have 5-10 headers)
    let mut forwarded = Map::with_capacity(headers.len().min(10));

    for (name, value) in headers.iter() {
        if CHAT_COMPLETIONS_HEADERS_NOT_FORWARDED
            .iter()
            .any(|excluded| name.as_str().eq_ignore_ascii_case(excluded))
        {
            continue;
        }

        let value = value.to_str().map_err(|_| {
            CoreError::InvalidRequest(format!("invalid value for header {}", name.as_str()))
        })?;
        forwarded.insert(name.to_string(), Value::String(value.to_string()));
    }

    Ok((!forwarded.is_empty()).then_some(forwarded))
}

#[derive(Debug)]
struct ChatCompletionsRouteError(CoreError);

impl From<CoreError> for ChatCompletionsRouteError {
    fn from(error: CoreError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ChatCompletionsRouteError {
    fn into_response(self) -> Response {
        let (status, message) = match self.0 {
            CoreError::InvalidRequest(message) => (StatusCode::BAD_REQUEST, message),
            CoreError::InvalidProvider(_) | CoreError::Routing(_) => (
                StatusCode::NOT_FOUND,
                "no chat completions deployment is configured for this model".to_string(),
            ),
            CoreError::Auth(message) => {
                if message.contains("budget") {
                    (StatusCode::PAYMENT_REQUIRED, message)
                } else if message.contains("rate limit") {
                    (StatusCode::TOO_MANY_REQUESTS, message)
                } else if message.contains("access") || message.contains("blocked") {
                    (StatusCode::FORBIDDEN, message)
                } else {
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("provider authentication failed: {message}"),
                    )
                }
            }
            CoreError::Http { .. }
            | CoreError::Network(_)
            | CoreError::Connect(_)
            | CoreError::InvalidResponse(_)
            | CoreError::InvalidType { .. }
            | CoreError::MissingField(_) => (
                StatusCode::BAD_GATEWAY,
                "chat completions provider request failed".to_string(),
            ),
            CoreError::Unsupported(reason) => (
                StatusCode::BAD_REQUEST,
                format!("chat completions request is not supported: {reason}"),
            ),
            CoreError::Timeout(message) => (
                StatusCode::GATEWAY_TIMEOUT,
                format!("chat completions request timed out: {message}"),
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

    use axum::Router;
    use axum::body::Body;
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

    use litellm_core::auth::KeyCache;
    use std::time::Duration;

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
                healthy: Some(true),
                weight: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            }])),
            master_key: master_key.map(Arc::from),
            loggers: Arc::new(Vec::new()),
            realtime_pool: RealtimePool::disabled(),
            key_cache: Arc::new(KeyCache::new(Duration::from_secs(600), 10_000)),
            redis: None,
            postgres: None,
            spend_worker: None,
            http_client: Arc::new(reqwest::Client::new()),
            circuit_breakers: Arc::new(crate::auth::circuit_breaker::CircuitBreakerRegistry::new(
                crate::auth::circuit_breaker::CircuitBreakerConfig::default(),
            )),
            metrics: Arc::new(crate::metrics::GatewayMetrics::new()),
            config: crate::state::GatewayConfig::from_env(),
            global_rate_limiter: Arc::new(crate::hardening::GlobalRateLimiter::new(10_000, 60)),
            secret_rotator: None,
            audit_log_shipper: None,
            csrf_state: Arc::new(crate::middleware::csrf::CsrfState::new(3600)),
            alerting_state: Arc::new(crate::middleware::alerting::AlertingState::new(
                crate::alerting::AlertingConfig::default(),
            )),
            guardrail_runner: Arc::new(
                crate::integrations::custom_guardrail::CustomGuardrailRunner::new(Vec::new()),
            ),
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
            let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-test","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}"#;
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

    #[tokio::test]
    async fn route_constructs_upstream_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let app = app(state("claude-test", api_base, Some("master-key")));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "claude-test",
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
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("response is json");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["usage"]["prompt_tokens"], 10);
        assert_eq!(parsed["usage"]["completion_tokens"], 5);
        let upstream_request = server.await.expect("upstream task completes");
        let (_, upstream_body) = upstream_request
            .split_once("\r\n\r\n")
            .expect("upstream request has body");
        let upstream_body: serde_json::Value =
            serde_json::from_str(upstream_body).expect("upstream body is json");
        assert_eq!(upstream_body["model"], "claude-test");
        let content = &upstream_body["messages"][0]["content"];
        if content.is_array() {
            assert_eq!(content[0]["text"], "hello");
        } else {
            assert_eq!(content.as_str().unwrap(), "hello");
        }
    }

    #[tokio::test]
    async fn route_substitutes_model_alias_with_provider_model() {
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
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "production",
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
                    .uri("/v1/chat/completions")
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
                    .uri("/v1/chat/completions")
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
    async fn route_rejects_malformed_json() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from("{not-json"))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn route_rejects_missing_model_field() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn route_returns_404_for_unconfigured_model() {
        let app = app(state(
            "claude-test",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "unknown-model",
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("error is json");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no chat completions deployment")
        );
    }

    #[tokio::test]
    async fn input_validation_rejects_missing_messages() {
        let app = Router::new()
            .merge(super::router())
            .with_state(state_with_provider(
                "gpt-4",
                "gpt-4",
                "http://unused".into(),
                Some("master-key"),
            ));

        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer master-key")
                .header("content-type", "application/json")
                .body(Body::from(json!({"model": "gpt-4"}).to_string()))
                .expect("request builds"),
        )
        .await
        .expect("route responds");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("messages")
        );
    }

    #[tokio::test]
    async fn input_validation_rejects_invalid_temperature() {
        let app = Router::new()
            .merge(super::router())
            .with_state(state_with_provider(
                "gpt-4",
                "gpt-4",
                "http://unused".into(),
                Some("master-key"),
            ));

        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer master-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "temperature": 5.0
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("route responds");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap()
                .contains("temperature")
        );
    }

    #[tokio::test]
    async fn budget_enforcement_allows_within_budget_key() {
        let app = Router::new()
            .merge(super::router())
            .with_state(state_with_provider(
                "gpt-4",
                "gpt-4",
                "http://unused".into(),
                Some("master-key"),
            ));

        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer master-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}]
                    })
                    .to_string(),
                ))
                .expect("request builds"),
        )
        .await
        .expect("route responds");

        // Key has no budget limit, so request should proceed (may fail on upstream, but not on budget)
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn metrics_endpoint_returns_prometheus_format() {
        let state =
            state_with_provider("gpt-4", "gpt-4", "http://unused".into(), Some("master-key"));
        let app = Router::new()
            .route("/metrics", axum::routing::get(crate::routes::metrics))
            .with_state(state);

        let unauthenticated = tower::ServiceExt::oneshot(
            app.clone(),
            axum::http::Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("route responds");

        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

        let response = tower::ServiceExt::oneshot(
            app,
            axum::http::Request::builder()
                .uri("/metrics")
                .header("authorization", "Bearer master-key")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("route responds");

        assert_eq!(response.status(), StatusCode::OK);
    }
}
