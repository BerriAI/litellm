//! Distributed tracing middleware using OpenTelemetry.
//!
//! Provides distributed tracing support for tracking requests across services.

use axum::{body::Body, extract::Request, http::header, middleware::Next, response::Response};
use std::time::Instant;
use uuid::Uuid;

/// Tracing configuration.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for tracing.
    pub service_name: String,
    /// Whether to enable tracing.
    pub enabled: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "litellm-gateway".to_string(),
            enabled: true,
        }
    }
}

/// Tracing middleware state.
#[allow(dead_code)]
pub struct TracingState {
    config: TracingConfig,
}

impl TracingState {
    /// Create a new tracing state.
    pub fn new(config: TracingConfig) -> Self {
        Self { config }
    }
}

/// Distributed tracing middleware.
pub async fn tracing_middleware(request: Request<Body>, next: Next) -> Response {
    let start = Instant::now();

    // Generate or extract trace ID
    let trace_id = request
        .headers()
        .get("X-Trace-ID")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // Generate span ID
    let span_id = Uuid::new_v4().to_string();

    // Extract request information
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let user_agent = request
        .headers()
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Log the start of the request
    tracing::info!(
        trace_id = %trace_id,
        span_id = %span_id,
        method = %method,
        path = %path,
        user_agent = ?user_agent,
        "Request started"
    );

    // Process the request
    let response = next.run(request).await;

    // Calculate duration
    let duration = start.elapsed();
    let status = response.status().as_u16();

    // Log the end of the request
    tracing::info!(
        trace_id = %trace_id,
        span_id = %span_id,
        status = status,
        duration_ms = duration.as_millis(),
        "Request completed"
    );

    // Add trace ID to response headers
    let mut response = response;
    response
        .headers_mut()
        .insert("X-Trace-ID", trace_id.parse().unwrap());

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use axum::routing::get;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_tracing_middleware() {
        let app = Router::new()
            .route("/test", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn(tracing_middleware));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("X-Trace-ID").is_some());
    }

    #[tokio::test]
    async fn test_tracing_preserves_trace_id() {
        let app = Router::new()
            .route("/test", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn(tracing_middleware));

        let trace_id = "test-trace-id-123";
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .header("X-Trace-ID", trace_id)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().get("X-Trace-ID").unwrap(), trace_id);
    }
}
