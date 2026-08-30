//! Comprehensive middleware integration tests.
//!
//! Tests middleware components individually and in compatible combinations.

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
        middleware::from_fn,
        routing::post,
        Router,
    };
    use tower::ServiceExt;
    use std::sync::Arc;
    use crate::middleware::{
        validation::validation_middleware,
        security_headers::security_headers_middleware,
        tracing::tracing_middleware,
    };

    async fn test_handler() -> &'static str {
        "OK"
    }

    #[tokio::test]
    async fn test_security_headers_middleware() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(security_headers_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("X-Frame-Options").unwrap(),
            "DENY"
        );
        assert_eq!(
            response.headers().get("X-Content-Type-Options").unwrap(),
            "nosniff"
        );
        assert_eq!(
            response.headers().get("X-XSS-Protection").unwrap(),
            "1; mode=block"
        );
        assert_eq!(
            response.headers().get("Strict-Transport-Security").unwrap(),
            "max-age=31536000; includeSubDomains"
        );
        assert_eq!(
            response.headers().get("Referrer-Policy").unwrap(),
            "strict-origin-when-cross-origin"
        );
    }

    #[tokio::test]
    async fn test_validation_middleware_large_payload() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(validation_middleware));

        let large_body = "x".repeat(20_000_000);
        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .header("content-length", large_body.len().to_string())
            .body(Body::from(large_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_validation_middleware_normal_request() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(validation_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .header("content-length", "100")
            .body(Body::from(r#"{"model": "gpt-4", "messages": []}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_tracing_middleware_generates_trace_id() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(tracing_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("X-Trace-ID").is_some());
    }

    #[tokio::test]
    async fn test_tracing_middleware_propagates_trace_id() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(tracing_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .header("X-Trace-ID", "custom-trace-id-123")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("X-Trace-ID").unwrap(),
            "custom-trace-id-123"
        );
    }

    #[tokio::test]
    async fn test_combined_security_and_tracing() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(tracing_middleware))
            .layer(from_fn(security_headers_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        
        // Check security headers
        assert_eq!(
            response.headers().get("X-Frame-Options").unwrap(),
            "DENY"
        );
        
        // Check tracing headers
        assert!(response.headers().get("X-Trace-ID").is_some());
    }

    #[tokio::test]
    async fn test_combined_validation_and_security() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(security_headers_middleware))
            .layer(from_fn(validation_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .header("content-length", "50")
            .body(Body::from(r#"{"model": "gpt-4", "messages": []}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("X-Frame-Options").unwrap(),
            "DENY"
        );
    }

    #[tokio::test]
    async fn test_metrics_recorded() {
        use crate::metrics::GatewayMetrics;

        let metrics = GatewayMetrics::new();

        metrics
            .validation_failures_total
            .with_label_values(&["payload_too_large"])
            .inc();

        metrics
            .csrf_rejections_total
            .with_label_values(&["missing_token"])
            .inc();

        metrics
            .cors_preflight_requests_total
            .with_label_values(&["https://example.com", "POST"])
            .inc();

        metrics
            .trace_spans_created_total
            .with_label_values(&["ai-gateway"])
            .inc();

        let output = metrics.render();
        assert!(output.contains("litellm_validation_failures_total"));
        assert!(output.contains("litellm_csrf_rejections_total"));
        assert!(output.contains("litellm_cors_preflight_requests_total"));
        assert!(output.contains("litellm_trace_spans_created_total"));
    }

    #[tokio::test]
    async fn test_alerting_state_tracking() {
        use crate::middleware::alerting::AlertingState;
        use crate::alerting::AlertingConfig;

        let alerting_state = AlertingState::new(AlertingConfig::default());

        alerting_state.record_request().await;
        alerting_state.record_request().await;
        alerting_state.record_error().await;

        let error_rate = alerting_state.get_error_rate().await;
        assert_eq!(error_rate, 0.5);

        alerting_state.reset_counters().await;
        let error_rate_after_reset = alerting_state.get_error_rate().await;
        assert_eq!(error_rate_after_reset, 0.0);
    }

    #[tokio::test]
    async fn test_csrf_state_token_generation() {
        use crate::middleware::csrf::CsrfState;

        let csrf_state = CsrfState::new(3600);
        let token = csrf_state.generate_token("session-123").await;

        assert!(!token.is_empty());
        assert!(token.len() > 20);
    }

    #[tokio::test]
    async fn test_full_middleware_stack() {
        let app = Router::new()
            .route("/test", post(test_handler))
            .layer(from_fn(tracing_middleware))
            .layer(from_fn(security_headers_middleware))
            .layer(from_fn(validation_middleware));

        let request = Request::builder()
            .method(Method::POST)
            .uri("/test")
            .header("content-length", "50")
            .body(Body::from(r#"{"model": "gpt-4", "messages": []}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        
        // Check security headers
        assert_eq!(
            response.headers().get("X-Frame-Options").unwrap(),
            "DENY"
        );
        
        // Check tracing headers
        assert!(response.headers().get("X-Trace-ID").is_some());
    }
}
