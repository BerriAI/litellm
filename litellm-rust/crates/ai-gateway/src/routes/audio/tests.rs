//! Tests for audio routes.

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use tower::ServiceExt;

    use crate::routes::app;
    use crate::state::AppState;
    use litellm_core::auth::KeyCache;
    use litellm_core::router::{Deployment, LiteLLMParams, Router};
    use std::sync::Arc;
    use std::time::Duration;

    fn test_state() -> AppState {
        AppState {
            router: Arc::new(Router::new(vec![Deployment {
                model_name: "tts-1".to_string(),
                litellm_params: LiteLLMParams {
                    model: "openai/tts-1".to_string(),
                    api_key: Some("test-key".to_string()),
                    api_base: None,
                },
                healthy: Some(true),
                weight: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            }])),
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
            alerting_state: Arc::new(crate::middleware::alerting::AlertingState::new(crate::alerting::AlertingConfig::default())),
            guardrail_runner: Arc::new(crate::integrations::custom_guardrail::CustomGuardrailRunner::new(vec![])),
        }
    }

    #[tokio::test]
    async fn test_speech_route_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "tts-1",
                            "input": "Hello world",
                            "voice": "alloy"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_speech_route_validates_model() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("authorization", "Bearer test-master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "input": "Hello world",
                            "voice": "alloy"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_speech_route_validates_input() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("authorization", "Bearer test-master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "tts-1",
                            "voice": "alloy"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_speech_route_validates_voice() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header("authorization", "Bearer test-master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "tts-1",
                            "input": "Hello world"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_transcription_route_requires_auth() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "whisper-1",
                            "file": "dGVzdA==" // base64 encoded "test"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_transcription_route_validates_model() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header("authorization", "Bearer test-master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "file": "dGVzdA=="
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_transcription_route_validates_file() {
        let app = app(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header("authorization", "Bearer test-master-key")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "whisper-1"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
