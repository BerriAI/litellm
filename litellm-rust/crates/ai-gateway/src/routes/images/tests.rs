#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use litellm_core::auth::KeyCache;
    use litellm_core::router::{Deployment, LiteLLMParams, Router as ModelRouter};
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    use super::super::super::app;
    use crate::io::realtime_pool::RealtimePool;
    use crate::state::AppState;

    fn state(model: &str, api_base: String, master_key: Option<&str>) -> AppState {
        AppState {
            router: Arc::new(ModelRouter::new(vec![Deployment {
                model_name: model.to_string(),
                litellm_params: LiteLLMParams {
                    model: format!("openai/{model}"),
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
            guardrail_runner: Arc::new(crate::integrations::custom_guardrail::CustomGuardrailRunner::new(Vec::new())),
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
            let body = r#"{"created":1589478378,"data":[{"url":"https://example.com/image.png"}]}"#;
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
    async fn route_constructs_openai_upstream_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let (api_base, server) = upstream(listener).await;
        let app = app(state("dall-e-3", api_base, Some("master-key")));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
                    .header("authorization", "Bearer master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "dall-e-3",
                            "prompt": "A cute baby sea otter",
                            "n": 1,
                            "size": "1024x1024"
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
            serde_json::from_slice::<serde_json::Value>(&body).expect("json")["created"],
            1589478378
        );
        let upstream_request = server.await.expect("upstream task completes");
        let (_, upstream_body) = upstream_request
            .split_once("\r\n\r\n")
            .expect("upstream request has body");
        let upstream_body: serde_json::Value =
            serde_json::from_str(upstream_body).expect("upstream body is json");
        assert_eq!(upstream_body["model"], "dall-e-3");
        assert_eq!(upstream_body["prompt"], "A cute baby sea otter");
        assert_eq!(upstream_body["n"], 1);
        assert_eq!(upstream_body["size"], "1024x1024");
    }

    #[tokio::test]
    async fn route_rejects_missing_master_key() {
        let app = app(state(
            "dall-e-3",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
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
            "dall-e-3",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
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
            "dall-e-3",
            "http://127.0.0.1:1".to_string(),
            Some("master-key"),
        ));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
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
