//! The authenticated-key identity (`UserApiKeyAuth`) plus the axum extractor that
//! resolves a bearer token into one.
//!
//! `UserApiKeyAuth` mirrors the subset of the Python proxy's `UserAPIKeyAuth`
//! object the gateway needs to enforce auth and (later) budgets/limits. It is the
//! value every authenticated handler receives.
//!
//! The extractor is the single place key resolution happens:
//! master key → synthetic admin; otherwise cache, then [`KeyAuthenticator`]. The
//! authenticator is a trait object on [`AppState`], so the "call Python" backend
//! can be swapped for a native Rust one WITHOUT touching this file or any route.

use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use axum::http::StatusCode;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::auth::AuthError;
use crate::state::AppState;

/// The authenticated identity behind a request, mirroring the fields the Python
/// proxy returns from key verification. All fields are optional except `spend`
/// and `models`, which the proxy always emits (default to `0.0` / empty here so a
/// partial JSON body still deserializes).
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct UserApiKeyAuth {
    pub api_key: Option<String>,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub user_role: Option<String>,
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub spend: f64,
    pub blocked: Option<bool>,
    #[serde(default)]
    pub models: Vec<String>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
}

impl UserApiKeyAuth {
    /// The synthetic identity granted to the master key: a `proxy_admin` with no
    /// budget/limit fields set. Constructed locally so presenting the master key
    /// never requires a round-trip to the authenticator.
    pub fn admin() -> Self {
        Self {
            user_role: Some("proxy_admin".to_string()),
            ..Self::default()
        }
    }
}

/// SHA-256 over `(route, model, key)`, used as the cache key. Hashing keeps
/// plaintext keys out of the in-memory cache map; including the route AND model
/// means an authorization cached for one (route, model) is never reused for a
/// different one — both can be restricted independently per key/team.
pub(crate) fn key_hash(key: &str, route: &str, model: Option<&str>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(route.as_bytes());
    hasher.update([0u8]); // domain separator
    hasher.update(model.unwrap_or("").as_bytes());
    hasher.update([0u8]);
    hasher.update(key.as_bytes());
    hasher.finalize().into()
}

/// Extract the trimmed bearer token from the `Authorization` header, if present.
fn bearer_token(parts: &Parts) -> Option<&str> {
    parts
        .headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

/// Pull a query parameter's value out of a raw query string, e.g. `model` from
/// `model=gpt-realtime&foo=bar`. Returns the raw (un-percent-decoded) value;
/// model names use only query-safe characters (alphanumerics, `-`, `_`, `.`, `/`),
/// so no decoding is needed for the values we authorize on.
fn query_param<'a>(query: Option<&'a str>, name: &str) -> Option<&'a str> {
    query?.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == name).then_some(v)
    })
}

#[axum::async_trait]
impl FromRequestParts<AppState> for UserApiKeyAuth {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let Some(token) = bearer_token(parts) else {
            return Err((
                StatusCode::UNAUTHORIZED,
                "missing or invalid bearer token".to_string(),
            ));
        };

        // Master key → synthetic admin, no authenticator round-trip. Constant-time
        // compare to avoid leaking the key via timing (mirrors `RequireMasterKey`).
        if state.is_master_key(token) {
            return Ok(Self::admin());
        }

        // Validate against the route AND requested model, so the backend enforces
        // the key's route/model restrictions for what's actually being requested
        // (not just that the key exists — closes the model-authorization bypass).
        let route = parts.uri.path();
        let model = query_param(parts.uri.query(), "model");

        // Virtual key: serve from cache, else verify via the (swappable) backend
        // and cache the result keyed by (route, model, key) hash.
        let hash = key_hash(token, route, model);
        if let Some(cached) = state.key_cache.get(&hash) {
            return Ok(cached);
        }

        match state.authenticator.verify(token, route, model).await {
            Ok(auth) => {
                state.key_cache.insert(hash, auth.clone());
                Ok(auth)
            }
            Err(AuthError::Unauthorized) => {
                Err((StatusCode::UNAUTHORIZED, "invalid api key".to_string()))
            }
            Err(AuthError::Upstream(detail)) => {
                // Keep the backend detail (which can include internal URLs/topology)
                // server-side only; return a generic message to the caller.
                eprintln!("auth backend error: {detail}");
                Err((
                    StatusCode::UNAUTHORIZED,
                    "authentication unavailable".to_string(),
                ))
            }
        }
    }
}

/// Constant-time compare of `token` against the configured master key, if any.
pub(crate) fn is_master_key(master_key: Option<&str>, token: &str) -> bool {
    match master_key {
        Some(expected) => bool::from(token.as_bytes().ct_eq(expected.as_bytes())),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{KeyAuthenticator, KeyCache};
    use crate::io::realtime_pool::RealtimePool;
    use axum::http::Request;
    use litellm_core::router::Router;
    use std::sync::Arc;

    /// Stub authenticator so tests never touch the network. Records nothing; just
    /// returns a fixed identity (or unauthorized) per construction.
    struct StubAuthenticator {
        result: Result<UserApiKeyAuth, AuthError>,
    }

    #[axum::async_trait]
    impl KeyAuthenticator for StubAuthenticator {
        async fn verify(
            &self,
            _key: &str,
            _route: &str,
            _model: Option<&str>,
        ) -> Result<UserApiKeyAuth, AuthError> {
            match &self.result {
                Ok(auth) => Ok(auth.clone()),
                Err(AuthError::Unauthorized) => Err(AuthError::Unauthorized),
                Err(AuthError::Upstream(detail)) => Err(AuthError::Upstream(detail.clone())),
            }
        }
    }

    fn test_state(master_key: Option<&str>, authenticator: Arc<dyn KeyAuthenticator>) -> AppState {
        AppState {
            router: Arc::new(Router::new(vec![])),
            master_key: master_key.map(Arc::from),
            realtime_pool: RealtimePool::disabled(),
            authenticator,
            key_cache: Arc::new(KeyCache::new()),
        }
    }

    async fn extract(state: &AppState, header: Option<&str>) -> Result<UserApiKeyAuth, StatusCode> {
        let mut builder = Request::builder().uri("/");
        if let Some(value) = header {
            builder = builder.header(AUTHORIZATION, value);
        }
        let request = builder.body(()).unwrap();
        let (mut parts, ()) = request.into_parts();
        UserApiKeyAuth::from_request_parts(&mut parts, state)
            .await
            .map_err(|(status, _)| status)
    }

    #[tokio::test]
    async fn master_key_yields_admin_without_calling_authenticator() {
        let stub = Arc::new(StubAuthenticator {
            // If the extractor reached the authenticator, it'd return this — but the
            // master-key branch should short-circuit before that.
            result: Err(AuthError::Unauthorized),
        });
        let state = test_state(Some("sk-master"), stub);

        let auth = extract(&state, Some("Bearer sk-master")).await.unwrap();
        assert_eq!(auth.user_role.as_deref(), Some("proxy_admin"));
    }

    #[tokio::test]
    async fn missing_token_is_unauthorized() {
        let stub = Arc::new(StubAuthenticator {
            result: Ok(UserApiKeyAuth::default()),
        });
        let state = test_state(Some("sk-master"), stub);

        let status = extract(&state, None).await.unwrap_err();
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn virtual_key_verifies_then_caches() {
        let stub = Arc::new(StubAuthenticator {
            result: Ok(UserApiKeyAuth {
                user_id: Some("u-1".to_string()),
                ..UserApiKeyAuth::default()
            }),
        });
        let state = test_state(Some("sk-master"), stub);

        let auth = extract(&state, Some("Bearer sk-virtual")).await.unwrap();
        assert_eq!(auth.user_id.as_deref(), Some("u-1"));
        // Second call should be served from cache (same identity).
        let cached = extract(&state, Some("Bearer sk-virtual")).await.unwrap();
        assert_eq!(cached.user_id.as_deref(), Some("u-1"));
    }

    #[tokio::test]
    async fn unauthorized_virtual_key_maps_to_401() {
        let stub = Arc::new(StubAuthenticator {
            result: Err(AuthError::Unauthorized),
        });
        let state = test_state(Some("sk-master"), stub);

        let status = extract(&state, Some("Bearer sk-bad")).await.unwrap_err();
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn deserializes_python_verify_body() {
        // Sample body shaped like the Python proxy's key-verification response.
        let body = serde_json::json!({
            "api_key": "hashed-abc",
            "key_alias": "my-key",
            "user_id": "user-123",
            "team_id": "team-456",
            "org_id": "org-789",
            "user_role": "internal_user",
            "max_budget": 100.0,
            "spend": 12.5,
            "blocked": false,
            "models": ["gpt-4o", "gpt-4o-mini"],
            "tpm_limit": 1000,
            "rpm_limit": 60
        });

        let auth: UserApiKeyAuth = serde_json::from_value(body).unwrap();
        assert_eq!(auth.api_key.as_deref(), Some("hashed-abc"));
        assert_eq!(auth.key_alias.as_deref(), Some("my-key"));
        assert_eq!(auth.user_id.as_deref(), Some("user-123"));
        assert_eq!(auth.team_id.as_deref(), Some("team-456"));
        assert_eq!(auth.org_id.as_deref(), Some("org-789"));
        assert_eq!(auth.user_role.as_deref(), Some("internal_user"));
        assert_eq!(auth.max_budget, Some(100.0));
        assert_eq!(auth.spend, 12.5);
        assert_eq!(auth.blocked, Some(false));
        assert_eq!(auth.models, vec!["gpt-4o", "gpt-4o-mini"]);
        assert_eq!(auth.tpm_limit, Some(1000));
        assert_eq!(auth.rpm_limit, Some(60));
    }

    #[test]
    fn deserializes_minimal_body_with_defaults() {
        // Only the always-present fields; spend defaults to 0.0, models to empty.
        let body = serde_json::json!({ "user_id": "u" });
        let auth: UserApiKeyAuth = serde_json::from_value(body).unwrap();
        assert_eq!(auth.spend, 0.0);
        assert!(auth.models.is_empty());
        assert!(auth.max_budget.is_none());
    }

    #[test]
    fn admin_is_proxy_admin() {
        assert_eq!(
            UserApiKeyAuth::admin().user_role.as_deref(),
            Some("proxy_admin")
        );
    }
}
