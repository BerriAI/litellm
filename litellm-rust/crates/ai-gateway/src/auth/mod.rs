//! Gateway authentication, as an axum **extractor** (the idiomatic pattern —
//! keeps handlers clean and auth testable).
//!
//! For now this is a single **master key**: any caller presenting it as
//! `Authorization: Bearer <key>` may invoke the gateway. Per-key auth, budgets,
//! and rate limits are delegated to the Python proxy in a later phase.
//!
//! A handler opts in by adding [`RequireMasterKey`] to its arguments; auth then
//! runs during extraction, before the handler body. Routes never re-implement it.

#[cfg(feature = "aws-auth")]
pub mod aws;
#[cfg(feature = "aws-auth")]
mod constants;

#[cfg(feature = "server")]
use axum::extract::FromRequestParts;
#[cfg(feature = "server")]
use axum::http::header::AUTHORIZATION;
#[cfg(feature = "server")]
use axum::http::request::Parts;
#[cfg(feature = "server")]
use axum::http::StatusCode;
#[cfg(feature = "server")]
use sha2::{Digest, Sha256};
#[cfg(feature = "server")]
use subtle::ConstantTimeEq;

#[cfg(feature = "server")]
use crate::state::AppState;

/// SHA-256 hex digest of a token — the exact transform the Python proxy applies
/// (`litellm.proxy.utils.hash_token`).
///
/// STRICT REQUIREMENT: a raw key (`LITELLM_MASTER_KEY`, a virtual key, …) must
/// **never** leave this gateway in a log payload. Spend logs and every callback
/// integration receive `user_api_key_hash`, so that field must be this hash, not
/// the credential. Hashing here also means the value matches the key's hash in
/// `LiteLLM_SpendLogs.api_key`, so realtime spend joins with the rest of LiteLLM.
#[cfg(feature = "server")]
pub fn hash_token(token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Extractor that requires the configured master key as a bearer token.
///
/// Rejections: `500` when no master key is configured (permanent
/// misconfiguration, not a transient outage); `401` on a missing/incorrect
/// token. The comparison is constant-time.
#[cfg(feature = "server")]
pub struct RequireMasterKey;

#[cfg(feature = "server")]
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

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::hash_token;

    #[test]
    fn hash_token_matches_python_sha256_hexdigest() {
        // Must equal hashlib.sha256("sk-1234".encode()).hexdigest() — the value
        // the proxy stores in LiteLLM_SpendLogs.api_key.
        assert_eq!(
            hash_token("sk-1234"),
            "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
        );
        // 64 lowercase hex chars, and never the raw input.
        let h = hash_token("sk-secret");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(h, "sk-secret");
    }
}
