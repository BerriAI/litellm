//! Gateway authentication, as an axum **extractor** (the idiomatic pattern —
//! keeps handlers clean and auth testable).
//!
//! For now this is a single **master key**: any caller presenting it as
//! `Authorization: Bearer <key>` may invoke the gateway. Per-key auth, budgets,
//! and rate limits are delegated to the Python proxy in a later phase.
//!
//! A handler opts in by adding [`RequireMasterKey`] to its arguments; auth then
//! runs during extraction, before the handler body. Routes never re-implement it.

pub mod cache;
pub mod client;
pub mod user_api_key;

pub use cache::KeyCache;
pub use client::{AuthError, KeyAuthenticator};
pub use user_api_key::UserApiKeyAuth;

use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use axum::http::StatusCode;
use subtle::ConstantTimeEq;

use crate::state::AppState;

/// Extractor that requires the configured master key as a bearer token.
///
/// Rejections: `500` when no master key is configured (permanent
/// misconfiguration, not a transient outage); `401` on a missing/incorrect
/// token. The comparison is constant-time.
pub struct RequireMasterKey;

#[axum::async_trait]
impl FromRequestParts<AppState> for RequireMasterKey {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let Some(expected) = state.master_key.as_deref() else {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "gateway auth not configured (set LITELLM_MASTER_KEY)".to_string(),
            ));
        };
        let provided = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim);
        match provided {
            Some(token) if bool::from(token.as_bytes().ct_eq(expected.as_bytes())) => Ok(Self),
            _ => Err((
                StatusCode::UNAUTHORIZED,
                "missing or invalid bearer token".to_string(),
            )),
        }
    }
}
