//! CORS (Cross-Origin Resource Sharing) configuration middleware.
//!
//! Provides configurable CORS support for cross-origin requests.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{Method, StatusCode, header},
    middleware::Next,
    response::Response,
};
use std::collections::HashSet;
use std::sync::Arc;

/// CORS configuration.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins.
    pub allowed_origins: HashSet<String>,
    /// Allowed methods.
    pub allowed_methods: HashSet<Method>,
    /// Allowed headers.
    pub allowed_headers: HashSet<String>,
    /// Exposed headers.
    pub exposed_headers: HashSet<String>,
    /// Whether to allow credentials.
    pub allow_credentials: bool,
    /// Max age for preflight cache in seconds.
    pub max_age: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        let mut allowed_origins = HashSet::new();
        allowed_origins.insert("*".to_string());

        let mut allowed_methods = HashSet::new();
        allowed_methods.insert(Method::GET);
        allowed_methods.insert(Method::POST);
        allowed_methods.insert(Method::PUT);
        allowed_methods.insert(Method::DELETE);
        allowed_methods.insert(Method::OPTIONS);

        let mut allowed_headers = HashSet::new();
        allowed_headers.insert("Content-Type".to_string());
        allowed_headers.insert("Authorization".to_string());
        allowed_headers.insert("X-CSRF-Token".to_string());

        let mut exposed_headers = HashSet::new();
        exposed_headers.insert("X-Request-ID".to_string());

        Self {
            allowed_origins,
            allowed_methods,
            allowed_headers,
            exposed_headers,
            allow_credentials: false,
            max_age: 3600,
        }
    }
}

/// CORS middleware state.
pub struct CorsState {
    config: CorsConfig,
}

impl CorsState {
    /// Create a new CORS state.
    pub fn new(config: CorsConfig) -> Self {
        Self { config }
    }

    /// Check if an origin is allowed.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.config.allowed_origins.contains("*") || self.config.allowed_origins.contains(origin)
    }

    /// Check if a method is allowed.
    pub fn is_method_allowed(&self, method: &Method) -> bool {
        self.config.allowed_methods.contains(method)
    }

    /// Check if a header is allowed.
    pub fn is_header_allowed(&self, header: &str) -> bool {
        self.config.allowed_headers.contains(header)
    }
}

/// CORS middleware.
pub async fn cors_middleware(
    State(state): State<Arc<CorsState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Handle preflight requests
    if request.method() == Method::OPTIONS {
        let mut response = Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        if let Some(origin) = &origin {
            if state.is_origin_allowed(origin) {
                response
                    .headers_mut()
                    .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin.parse().unwrap());
            }
        }

        let allowed_methods: Vec<String> = state
            .config
            .allowed_methods
            .iter()
            .map(|m| m.to_string())
            .collect();
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            allowed_methods.join(", ").parse().unwrap(),
        );

        let allowed_headers: Vec<String> = state.config.allowed_headers.iter().cloned().collect();
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            allowed_headers.join(", ").parse().unwrap(),
        );

        response.headers_mut().insert(
            header::ACCESS_CONTROL_MAX_AGE,
            state.config.max_age.to_string().parse().unwrap(),
        );

        if state.config.allow_credentials {
            response.headers_mut().insert(
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                "true".parse().unwrap(),
            );
        }

        return Ok(response);
    }

    // Process the request
    let mut response = next.run(request).await;

    // Add CORS headers to the response
    if let Some(origin) = &origin {
        if state.is_origin_allowed(origin) {
            response
                .headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin.parse().unwrap());
        }
    }

    if state.config.allow_credentials {
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
            "true".parse().unwrap(),
        );
    }

    let exposed_headers: Vec<String> = state.config.exposed_headers.iter().cloned().collect();
    if !exposed_headers.is_empty() {
        response.headers_mut().insert(
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            exposed_headers.join(", ").parse().unwrap(),
        );
    }

    Ok(response)
}

/// Stateless CORS middleware using default configuration.
/// Suitable for the production router where per-instance state is not needed.
pub async fn cors_middleware_default(
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let config = CorsConfig::default();

    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    if request.method() == Method::OPTIONS {
        let mut response = Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        if let Some(origin) = &origin {
            let origin_allowed =
                config.allowed_origins.contains("*") || config.allowed_origins.contains(origin);
            if origin_allowed {
                response
                    .headers_mut()
                    .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin.parse().unwrap());
            }
        }

        let allowed_methods: Vec<String> = config
            .allowed_methods
            .iter()
            .map(|m| m.to_string())
            .collect();
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            allowed_methods.join(", ").parse().unwrap(),
        );

        let allowed_headers: Vec<String> = config.allowed_headers.iter().cloned().collect();
        response.headers_mut().insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            allowed_headers.join(", ").parse().unwrap(),
        );

        response.headers_mut().insert(
            header::ACCESS_CONTROL_MAX_AGE,
            config.max_age.to_string().parse().unwrap(),
        );

        return Ok(response);
    }

    let mut response = next.run(request).await;

    if let Some(origin) = &origin {
        let origin_allowed =
            config.allowed_origins.contains("*") || config.allowed_origins.contains(origin);
        if origin_allowed {
            response
                .headers_mut()
                .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin.parse().unwrap());
        }
    }

    let exposed_headers: Vec<String> = config.exposed_headers.iter().cloned().collect();
    if !exposed_headers.is_empty() {
        response.headers_mut().insert(
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            exposed_headers.join(", ").parse().unwrap(),
        );
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.allowed_origins.contains("*"));
        assert!(config.allowed_methods.contains(&Method::GET));
        assert!(config.allowed_methods.contains(&Method::POST));
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age, 3600);
    }

    #[test]
    fn test_is_origin_allowed() {
        let config = CorsConfig::default();
        let state = CorsState::new(config);

        assert!(state.is_origin_allowed("http://example.com"));
        assert!(state.is_origin_allowed("https://example.com"));
    }

    #[test]
    fn test_is_method_allowed() {
        let config = CorsConfig::default();
        let state = CorsState::new(config);

        assert!(state.is_method_allowed(&Method::GET));
        assert!(state.is_method_allowed(&Method::POST));
        assert!(!state.is_method_allowed(&Method::PATCH));
    }

    #[test]
    fn test_is_header_allowed() {
        let config = CorsConfig::default();
        let state = CorsState::new(config);

        assert!(state.is_header_allowed("Content-Type"));
        assert!(state.is_header_allowed("Authorization"));
        assert!(!state.is_header_allowed("X-Custom-Header"));
    }
}
