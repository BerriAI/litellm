//! Gateway authentication.
//!
//! For now this is a single **master key**: any caller presenting it as a bearer
//! token may invoke the gateway. There is no per-key auth, budgets, or rate
//! limits here — those are delegated to the Python proxy in a later phase.
//!
//! Routes import [`authorize`] and call it before doing any work, so the auth
//! policy lives in one place rather than being re-implemented per route.

use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderMap, StatusCode};
use subtle::ConstantTimeEq;

use crate::state::AppState;

/// Require the configured master key as an `Authorization: Bearer <key>` token.
///
/// Fails closed: with no master key configured the gateway rejects every request
/// with `500` (a permanent misconfiguration, not a transient outage). A
/// missing/incorrect token is `401`. The comparison is constant-time.
pub fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    let Some(expected) = state.master_key.as_deref() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "gateway auth not configured (set LITELLM_MASTER_KEY)".to_string(),
        ));
    };
    let provided = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    match provided {
        Some(token) if bool::from(token.as_bytes().ct_eq(expected.as_bytes())) => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".to_string(),
        )),
    }
}
