//! CSRF (Cross-Site Request Forgery) protection middleware.
//!
//! Provides token-based CSRF protection for state-changing operations.

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use base64::{Engine as _, engine::general_purpose};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// CSRF protection state.
#[allow(dead_code)]
pub struct CsrfState {
    /// Active CSRF tokens mapped to session IDs.
    tokens: Arc<Mutex<HashMap<String, String>>>,
    /// Token expiration duration in seconds.
    token_ttl: u64,
}

impl CsrfState {
    /// Create a new CSRF state.
    pub fn new(token_ttl: u64) -> Self {
        Self {
            tokens: Arc::new(Mutex::new(HashMap::new())),
            token_ttl,
        }
    }

    /// Generate a new CSRF token.
    pub async fn generate_token(&self, session_id: &str) -> String {
        let mut rng = rand::thread_rng();
        let token_bytes: [u8; 32] = rng.r#gen();
        let token = general_purpose::STANDARD.encode(token_bytes);

        let mut tokens = self.tokens.lock().await;
        tokens.insert(session_id.to_string(), token.clone());

        token
    }

    /// Validate a CSRF token.
    pub async fn validate_token(&self, session_id: &str, token: &str) -> bool {
        let tokens = self.tokens.lock().await;
        if let Some(stored_token) = tokens.get(session_id) {
            stored_token == token
        } else {
            false
        }
    }

    /// Remove a CSRF token.
    pub async fn remove_token(&self, session_id: &str) {
        let mut tokens = self.tokens.lock().await;
        tokens.remove(session_id);
    }
}

/// CSRF protection middleware.
pub async fn csrf_middleware(
    State(state): State<Arc<CsrfState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip CSRF validation for API routes that use bearer token authentication
    // CSRF protection is mainly needed for cookie-based authentication in browsers
    let path = request.uri().path();
    if path.starts_with("/v1/") || request.headers().get("Authorization").is_some() {
        return Ok(next.run(request).await);
    }

    // Only validate CSRF token for state-changing methods
    if matches!(
        request.method(),
        &axum::http::Method::POST
            | &axum::http::Method::PUT
            | &axum::http::Method::DELETE
            | &axum::http::Method::PATCH
    ) {
        // Get session ID from cookie or header
        let session_id = request
            .headers()
            .get("X-Session-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("default");

        // Get CSRF token from header
        let csrf_token = request
            .headers()
            .get("X-CSRF-Token")
            .and_then(|v| v.to_str().ok());

        if let Some(token) = csrf_token {
            if !state.validate_token(session_id, token).await {
                return Err(StatusCode::FORBIDDEN);
            }
        } else {
            // No CSRF token provided for state-changing request
            return Err(StatusCode::FORBIDDEN);
        }
    }

    Ok(next.run(request).await)
}

/// CSRF middleware that extracts CsrfState from AppState.
/// For use in the production router where AppState is the router state.
pub async fn csrf_middleware_from_app_state(
    State(state): State<crate::state::AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if matches!(
        request.method(),
        &axum::http::Method::POST
            | &axum::http::Method::PUT
            | &axum::http::Method::DELETE
            | &axum::http::Method::PATCH
    ) {
        // Skip CSRF validation for API routes that use bearer token authentication
        // CSRF protection is mainly needed for cookie-based authentication in browsers
        let path = request.uri().path();
        if path.starts_with("/v1/") || request.headers().get("Authorization").is_some() {
            return Ok(next.run(request).await);
        }

        let session_id = request
            .headers()
            .get("X-Session-ID")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("default");

        let csrf_token = request
            .headers()
            .get("X-CSRF-Token")
            .and_then(|v| v.to_str().ok());

        if let Some(token) = csrf_token {
            if !state.csrf_state.validate_token(session_id, token).await {
                return Err(StatusCode::FORBIDDEN);
            }
        } else {
            return Err(StatusCode::FORBIDDEN);
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_csrf_token_generation() {
        let state = CsrfState::new(3600);
        let token = state.generate_token("session1").await;

        assert!(!token.is_empty());
        assert!(state.validate_token("session1", &token).await);
    }

    #[tokio::test]
    async fn test_csrf_token_validation() {
        let state = CsrfState::new(3600);
        let token = state.generate_token("session1").await;

        assert!(state.validate_token("session1", &token).await);
        assert!(!state.validate_token("session2", &token).await);
        assert!(!state.validate_token("session1", "invalid_token").await);
    }

    #[tokio::test]
    async fn test_csrf_token_removal() {
        let state = CsrfState::new(3600);
        let token = state.generate_token("session1").await;

        assert!(state.validate_token("session1", &token).await);
        state.remove_token("session1").await;
        assert!(!state.validate_token("session1", &token).await);
    }
}
