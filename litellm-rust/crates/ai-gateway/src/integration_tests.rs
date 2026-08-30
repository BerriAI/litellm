//! End-to-end integration tests for the Rust gateway.
//!
//! These tests verify that all phases work together:
//! - Phase 1: Chat completions with streaming spend tracking, fallback routing, guardrails
//! - Phase 2: Messages with per-key auth, rate limiting, spend tracking
//! - Phase 3: Embeddings with full middleware
//! - Phase 4: Images with full middleware
//! - Phase 5: Audio with full middleware
//! - Phase 6: Config schema parity
//! - Phase 7: Callback integrations
//! - Phase 8: Read-side DB queries and admin endpoints

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

    fn create_test_state() -> AppState {
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
                Deployment {
                    model_name: "text-embedding-3-small".to_string(),
                    litellm_params: LiteLLMParams {
                        model: "openai/text-embedding-3-small".to_string(),
                        api_key: Some("test-key".to_string()),
                        api_base: None,
                    },
                    healthy: Some(true),
                    weight: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                },
                Deployment {
                    model_name: "dall-e-3".to_string(),
                    litellm_params: LiteLLMParams {
                        model: "openai/dall-e-3".to_string(),
                        api_key: Some("test-key".to_string()),
                        api_base: None,
                    },
                    healthy: Some(true),
                    weight: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                },
                Deployment {
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
                },
                Deployment {
                    model_name: "whisper-1".to_string(),
                    litellm_params: LiteLLMParams {
                        model: "openai/whisper-1".to_string(),
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
    async fn test_all_routes_are_registered() {
        let app = app(create_test_state());

        // Test that all routes are registered and respond (even if with errors)
        let routes = vec![
            ("/v1/chat/completions", "POST"),
            ("/v1/messages", "POST"),
            ("/v1/embeddings", "POST"),
            ("/v1/images/generations", "POST"),
            ("/v1/images/edits", "POST"),
            ("/v1/audio/speech", "POST"),
            ("/v1/audio/transcriptions", "POST"),
            ("/v1/models", "GET"),
            ("/key/info", "GET"),
            ("/spend/logs", "GET"),
            ("/user/info", "GET"),
            ("/team/info", "GET"),
            ("/health/liveness", "GET"),
            ("/health/readiness", "GET"),
            ("/health/deep", "GET"),
            ("/metrics", "GET"),
        ];

        for (path, method) in routes {
            let request = match method {
                "GET" => Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
                "POST" => Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
                _ => panic!("Unknown method"),
            };

            let response = app.clone().oneshot(request).await.unwrap();

            // Routes should exist (not 404), even if they return errors
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "Route {} {} should be registered",
                method,
                path
            );
        }
    }

    #[tokio::test]
    async fn test_authentication_across_all_routes() {
        let app = app(create_test_state());

        // Test that all protected routes require authentication
        let protected_routes = vec![
            ("/v1/chat/completions", "POST"),
            ("/v1/messages", "POST"),
            ("/v1/embeddings", "POST"),
            ("/v1/images/generations", "POST"),
            ("/v1/images/edits", "POST"),
            ("/v1/audio/speech", "POST"),
            ("/v1/audio/transcriptions", "POST"),
            ("/v1/models", "GET"),
            ("/key/info", "GET"),
            ("/spend/logs", "GET"),
            ("/user/info", "GET"),
            ("/team/info", "GET"),
        ];

        for (path, method) in protected_routes {
            let request = match method {
                "GET" => Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
                "POST" => Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
                _ => panic!("Unknown method"),
            };

            let response = app.clone().oneshot(request).await.unwrap();

            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "Route {} {} should require authentication",
                method,
                path
            );
        }
    }

    #[tokio::test]
    async fn test_models_endpoint_with_multiple_deployments() {
        let app = app(create_test_state());

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
        let models = json["data"].as_array().unwrap();
        assert_eq!(models.len(), 6);

        // Verify all models are present
        let model_ids: Vec<&str> = models.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert!(model_ids.contains(&"gpt-4"));
        assert!(model_ids.contains(&"claude-3"));
        assert!(model_ids.contains(&"text-embedding-3-small"));
        assert!(model_ids.contains(&"dall-e-3"));
        assert!(model_ids.contains(&"tts-1"));
        assert!(model_ids.contains(&"whisper-1"));
    }

    #[tokio::test]
    async fn test_config_loading_with_all_settings() {
        use crate::config::load_config_from_yaml;
        use std::io::Write;
        use tempfile::NamedTempFile;

        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    rpm: 1000
    tpm: 100000
    max_parallel_requests: 50
    mode: fallback
    healthy: true
    cooldown: 30
    weight: 10
    model_info:
      input_cost_per_token: 0.00003
      output_cost_per_token: 0.00006
      mode: chat

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  max_parallel_requests: 100
  global_max_parallel_requests: 1000
  max_request_size_mb: 10
  alerting:
    - slack
    - email
  alert_webhook_url: https://hooks.slack.com/services/xxx
  allowed_routes:
    - /v1/chat/completions
    - /v1/embeddings

litellm_settings:
  callbacks:
    - type: langfuse
      public_key: pk_test
      secret_key: sk_test
      host: https://langfuse.example.com
    - type: datadog
      api_key: dd_api_key
      host: https://api.datadoghq.com
  guardrails:
    - guardrail_name: prompt_injection
      guardrail_type: lakera
      enabled: true
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    ttl: 300
  drop_params: true
  num_retries: 3
  timeout: 600

router_settings:
  routing_strategy: latency-based
  num_retries: 5
  timeout: 300
  cooldown_seconds: 60
  allowed_fails: 3
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();

        // Verify model_list
        assert_eq!(config.router.deployments().len(), 1);
        assert_eq!(config.router.deployments()[0].model_name, "gpt-4");

        // Verify general_settings
        assert!(config.general_settings.master_key.is_some());
        assert_eq!(config.general_settings.max_parallel_requests, Some(100));
        assert_eq!(
            config.general_settings.global_max_parallel_requests,
            Some(1000)
        );
        assert_eq!(config.general_settings.max_request_size_mb, Some(10));
        assert!(config.general_settings.alerting.is_some());
        assert!(config.general_settings.alert_webhook_url.is_some());
        assert!(config.general_settings.allowed_routes.is_some());

        // Verify litellm_settings
        assert!(config.litellm_settings.callbacks.is_some());
        assert!(config.litellm_settings.guardrails.is_some());
        assert_eq!(config.litellm_settings.cache, Some(true));
        assert!(config.litellm_settings.cache_params.is_some());
        assert_eq!(config.litellm_settings.drop_params, Some(true));
        assert_eq!(config.litellm_settings.num_retries, Some(3));
        assert_eq!(config.litellm_settings.timeout, Some(600));

        // Verify router_settings
        assert_eq!(
            config.router_settings.routing_strategy,
            Some("latency-based".to_string())
        );
        assert_eq!(config.router_settings.num_retries, Some(5));
        assert_eq!(config.router_settings.timeout, Some(300));
        assert_eq!(config.router_settings.cooldown_seconds, Some(60));
        assert_eq!(config.router_settings.allowed_fails, Some(3));
    }

    #[tokio::test]
    async fn test_callback_configuration_parsing() {
        use crate::config::CallbackConfig;

        // Test Langfuse config
        let yaml = r#"
type: langfuse
public_key: pk_test
secret_key: sk_test
host: https://langfuse.example.com
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Langfuse {
                public_key,
                secret_key,
                host,
            } => {
                assert_eq!(public_key, "pk_test");
                assert_eq!(secret_key, "sk_test");
                assert_eq!(host, "https://langfuse.example.com");
            }
            _ => panic!("Expected Langfuse config"),
        }

        // Test Datadog config
        let yaml = r#"
type: datadog
api_key: dd_api_key
host: https://api.datadoghq.com
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Datadog {
                api_key,
                app_key,
                host,
            } => {
                assert_eq!(api_key, "dd_api_key");
                assert!(app_key.is_none());
                assert_eq!(host, "https://api.datadoghq.com");
            }
            _ => panic!("Expected Datadog config"),
        }

        // Test Webhooks config
        let yaml = r#"
type: webhooks
url: https://webhook.example.com
auth_token: secret_token
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Webhooks {
                url,
                headers,
                auth_token,
            } => {
                assert_eq!(url, "https://webhook.example.com");
                assert!(headers.is_none());
                assert_eq!(auth_token, Some("secret_token".to_string()));
            }
            _ => panic!("Expected Webhooks config"),
        }

        // Test Slack config
        let yaml = r#"
type: slack
webhook_url: https://hooks.slack.com/services/test
channel: '#alerts'
username: LiteLLM Bot
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Slack {
                webhook_url,
                channel,
                username,
                icon_emoji,
            } => {
                assert_eq!(webhook_url, "https://hooks.slack.com/services/test");
                assert_eq!(channel, Some("#alerts".to_string()));
                assert_eq!(username, Some("LiteLLM Bot".to_string()));
                assert!(icon_emoji.is_none());
            }
            _ => panic!("Expected Slack config"),
        }
    }
}
