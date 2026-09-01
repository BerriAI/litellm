//! Tests for admin endpoints.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::routes::app;
    use crate::state::AppState;
    use litellm_core::auth::KeyCache;
    use litellm_core::router::{Deployment, LiteLLMParams, Router};
    use std::sync::Arc;
    use std::time::Duration;

    fn test_state() -> AppState {
        AppState {
            router: Arc::new(Router::new(vec![
                Deployment {
                    model_name: "gpt-4".to_string(),
                    litellm_params: LiteLLMParams {
                        model: "openai/gpt-4".to_string(),
                        api_key: Some("test-key".to_string()),
                        api_base: None,
                    },
                    healthy: Some(true),
                    weight: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                },
                Deployment {
                    model_name: "claude-3".to_string(),
                    litellm_params: LiteLLMParams {
                        model: "anthropic/claude-3".to_string(),
                        api_key: Some("test-key".to_string()),
                        api_base: None,
                    },
                    healthy: Some(true),
                    weight: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                },
            ])),
            master_key: Some(Arc::from("test-master-key")),
            loggers: Arc::new(vec![]),
            realtime_pool: crate::io::realtime_pool::RealtimePool::disabled(),
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
                crate::integrations::custom_guardrail::CustomGuardrailRunner::new(vec![]),
            ),
        }
    }

    #[tokio::test]
    async fn test_models_endpoint_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_models_endpoint_returns_models() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .header("authorization", "Bearer test-master-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["data"][0]["id"], "gpt-4");
        assert_eq!(json["data"][1]["id"], "claude-3");
    }

    #[tokio::test]
    async fn test_key_info_endpoint_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/key/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_key_info_endpoint_returns_key_info() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/key/info")
                    .header("authorization", "Bearer test-master-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(json.get("token").is_some());
    }

    #[tokio::test]
    async fn test_spend_logs_endpoint_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/spend/logs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_spend_logs_endpoint_returns_empty_without_db() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/spend/logs")
                    .header("authorization", "Bearer test-master-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_user_info_endpoint_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/user/info?user_id=test-user")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_user_info_endpoint_returns_not_implemented_without_db() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/user/info?user_id=test-user")
                    .header("authorization", "Bearer test-master-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn test_team_info_endpoint_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/team/info?team_id=test-team")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_team_info_endpoint_returns_not_implemented_without_db() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/team/info?team_id=test-team")
                    .header("authorization", "Bearer test-master-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
