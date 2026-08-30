//! Security headers middleware.
//!
//! Adds security headers to all responses to protect against common web vulnerabilities.

use axum::{
    body::Body,
    extract::Request,
    http::header,
    middleware::Next,
    response::Response,
};

/// Security headers configuration.
#[derive(Debug, Clone)]
pub struct SecurityHeadersConfig {
    /// X-Frame-Options header value.
    pub x_frame_options: String,
    /// X-Content-Type-Options header value.
    pub x_content_type_options: String,
    /// X-XSS-Protection header value.
    pub x_xss_protection: String,
    /// Strict-Transport-Security header value.
    pub strict_transport_security: String,
    /// Content-Security-Policy header value.
    pub content_security_policy: String,
    /// Referrer-Policy header value.
    pub referrer_policy: String,
    /// Permissions-Policy header value.
    pub permissions_policy: String,
}

impl Default for SecurityHeadersConfig {
    fn default() -> Self {
        Self {
            x_frame_options: "DENY".to_string(),
            x_content_type_options: "nosniff".to_string(),
            x_xss_protection: "1; mode=block".to_string(),
            strict_transport_security: "max-age=31536000; includeSubDomains".to_string(),
            content_security_policy: "default-src 'self'".to_string(),
            referrer_policy: "strict-origin-when-cross-origin".to_string(),
            permissions_policy: "geolocation=(), microphone=(), camera=()".to_string(),
        }
    }
}

/// Security headers middleware.
pub async fn security_headers_middleware(
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;

    // Add security headers
    response.headers_mut().insert(
        header::X_FRAME_OPTIONS,
        "DENY".parse().unwrap(),
    );

    response.headers_mut().insert(
        "X-Content-Type-Options",
        "nosniff".parse().unwrap(),
    );

    response.headers_mut().insert(
        "X-XSS-Protection",
        "1; mode=block".parse().unwrap(),
    );

    response.headers_mut().insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );

    response.headers_mut().insert(
        "Content-Security-Policy",
        "default-src 'self'".parse().unwrap(),
    );

    response.headers_mut().insert(
        "Referrer-Policy",
        "strict-origin-when-cross-origin".parse().unwrap(),
    );

    response.headers_mut().insert(
        "Permissions-Policy",
        "geolocation=(), microphone=(), camera=()".parse().unwrap(),
    );

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::routing::get;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_security_headers_added() {
        let app = Router::new()
            .route("/test", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn(security_headers_middleware));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::X_FRAME_OPTIONS).unwrap(),
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
            response.headers().get("Content-Security-Policy").unwrap(),
            "default-src 'self'"
        );
        assert_eq!(
            response.headers().get("Referrer-Policy").unwrap(),
            "strict-origin-when-cross-origin"
        );
        assert_eq!(
            response.headers().get("Permissions-Policy").unwrap(),
            "geolocation=(), microphone=(), camera=()"
        );
    }
}
