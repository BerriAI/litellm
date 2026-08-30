//! Full per-key authentication extractor.
//!
//! Zero-allocation design: `extract_raw_key` borrows from headers, `HashedToken`
//! is stack-allocated, `KeyCache` returns `Arc<KeyObject>` (cheap clone).

use std::sync::Arc;

use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use litellm_core::auth::{HashedToken, KeyCache, KeyObject};
use subtle::ConstantTimeEq;

use crate::state::AppState;

/// Header names checked for the API key, in priority order.
const KEY_HEADERS: &[&str] = &["x-litellm-key", "api-key", "x-api-key", "x-google-api-key"];

/// Resolved authentication for a request. Zero-alloc: borrows from headers,
/// uses stack-allocated hash, and Arc-shared key object.
pub struct RequireValidKey {
    /// The resolved key object (from cache or DB). Arc-shared for zero-clone reads.
    pub key_object: Arc<KeyObject>,
    /// The SHA-256 hashed token. Stack-allocated, no heap.
    pub hashed_token: HashedToken,
}

/// Auth rejection reasons.
#[derive(Debug)]
pub enum AuthRejection {
    NotConfigured,
    KeyNotFound,
    Blocked,
    Expired,
    BackendError(String),
}

impl AuthRejection {
    pub fn status_code(&self) -> StatusCode {
        match self {
            AuthRejection::NotConfigured => StatusCode::INTERNAL_SERVER_ERROR,
            AuthRejection::KeyNotFound => StatusCode::UNAUTHORIZED,
            AuthRejection::Blocked => StatusCode::FORBIDDEN,
            AuthRejection::Expired => StatusCode::UNAUTHORIZED,
            AuthRejection::BackendError(_) => StatusCode::BAD_GATEWAY,
        }
    }

    pub fn message(&self) -> String {
        match self {
            AuthRejection::NotConfigured => {
                "gateway auth not configured (set LITELLM_MASTER_KEY)".to_string()
            }
            AuthRejection::KeyNotFound => "invalid or unknown API key".to_string(),
            AuthRejection::Blocked => "API key is blocked".to_string(),
            AuthRejection::Expired => "API key has expired".to_string(),
            AuthRejection::BackendError(msg) => format!("auth backend error: {msg}"),
        }
    }
}

impl From<AuthRejection> for (StatusCode, String) {
    fn from(rejection: AuthRejection) -> Self {
        (rejection.status_code(), rejection.message())
    }
}

/// Extract the raw API key from request headers. Zero-alloc: borrows from Parts.
fn extract_raw_key<'a>(parts: &'a Parts) -> Option<&'a str> {
    // Check Authorization header first (standard HTTP auth)
    if let Some(auth) = parts.headers.get(AUTHORIZATION) {
        if let Ok(value) = auth.to_str() {
            let trimmed = value.trim();
            for prefix in &["Bearer ", "bearer ", "Basic "] {
                if let Some(token) = trimmed.strip_prefix(prefix) {
                    let token = token.trim();
                    if !token.is_empty() {
                        return Some(token);
                    }
                }
            }
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }

    // Then check custom key headers
    for &header_name in KEY_HEADERS {
        if let Some(value) = parts.headers.get(header_name) {
            if let Ok(s) = value.to_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed);
                }
            }
        }
    }

    None
}

/// Constant-time master key comparison.
fn is_master_key(provided: &str, master_key: Option<&str>) -> bool {
    match master_key {
        Some(expected) => bool::from(provided.as_bytes().ct_eq(expected.as_bytes())),
        None => false,
    }
}

/// Static master key object. Allocated once at first access, never again.
fn master_key_object() -> Arc<KeyObject> {
    use std::sync::LazyLock;
    static MASTER: LazyLock<Arc<KeyObject>> = LazyLock::new(|| {
        Arc::new(KeyObject {
            token: "master".to_string(),
            key_name: Some("master_key".to_string()),
            key_alias: None,
            user_id: None,
            team_id: None,
            org_id: None,
            project_id: None,
            agent_id: None,
            spend: 0.0,
            max_budget: None,
            budget_duration: None,
            models: std::collections::HashSet::new(),
            tpm_limit: None,
            rpm_limit: None,
            max_parallel_requests: None,
            blocked: false,
            allowed_routes: std::collections::HashSet::new(),
            metadata: None,
            last_refreshed_at: None,
            expires: None,
        })
    });
    Arc::clone(&MASTER)
}

/// Look up a key in the cache. Returns Arc (zero-clone).
fn lookup_cache(cache: &KeyCache, hashed: &HashedToken) -> Option<Arc<KeyObject>> {
    let key_obj = cache.get(hashed)?;
    if key_obj.is_expired() {
        cache.remove(hashed);
        None
    } else {
        Some(key_obj)
    }
}

/// Look up a key in the database. Returns Arc-wrapped KeyObject.
async fn lookup_db(
    state: &AppState,
    hashed_token: &HashedToken,
) -> Result<Option<Arc<KeyObject>>, AuthRejection> {
    let Some(ref postgres) = state.postgres else {
        return Ok(None);
    };

    let row = sqlx::query_as::<_, KeyRow>(
        r#"SELECT token, key_name, user_id, team_id, org_id, spend, max_budget,
                  models, tpm_limit, rpm_limit, blocked, expires, allowed_routes
           FROM "LiteLLM_VerificationToken"
           WHERE token = $1"#,
    )
    .bind(hashed_token.as_hex_str())
    .fetch_optional(postgres.pool())
    .await
    .map_err(|e| AuthRejection::BackendError(format!("DB query failed: {e}")))?;

    Ok(row.map(|r| Arc::new(r.into_key_object())))
}

#[derive(sqlx::FromRow)]
struct KeyRow {
    token: String,
    key_name: Option<String>,
    user_id: Option<String>,
    team_id: Option<String>,
    org_id: Option<String>,
    spend: f64,
    max_budget: Option<f64>,
    models: Vec<String>,
    tpm_limit: Option<i64>,
    rpm_limit: Option<i64>,
    blocked: Option<bool>,
    expires: Option<String>,
    allowed_routes: Vec<String>,
}

impl KeyRow {
    fn into_key_object(self) -> KeyObject {
        KeyObject {
            token: self.token,
            key_name: self.key_name,
            key_alias: None,
            user_id: self.user_id,
            team_id: self.team_id,
            org_id: self.org_id,
            project_id: None,
            agent_id: None,
            spend: self.spend,
            max_budget: self.max_budget,
            budget_duration: None,
            models: self.models.into_iter().collect(),
            tpm_limit: self.tpm_limit,
            rpm_limit: self.rpm_limit,
            max_parallel_requests: None,
            blocked: self.blocked.unwrap_or(false),
            allowed_routes: self.allowed_routes.into_iter().collect(),
            metadata: None,
            last_refreshed_at: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            ),
            expires: self.expires,
        }
    }
}

#[axum::async_trait]
impl FromRequestParts<AppState> for RequireValidKey {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        // 1. Extract raw key (zero-alloc borrow from headers)
        let raw_key = extract_raw_key(parts).ok_or_else(|| {
            tracing::warn!("Auth: No key extracted from headers");
            AuthRejection::KeyNotFound
        })?;
        
        tracing::info!("Auth: Extracted raw_key='{}', master_key={:?}", raw_key, state.master_key.as_deref());

        // 2. Check master key (constant-time)
        if is_master_key(raw_key, state.master_key.as_deref()) {
            tracing::info!("Auth: Master key match!");
            return Ok(RequireValidKey {
                key_object: master_key_object(),
                hashed_token: HashedToken::hash(raw_key),
            });
        }
        tracing::warn!("Auth: Master key mismatch, proceeding to cache/DB lookup");

        // 3. Hash the key (stack-allocated)
        let hashed_token = HashedToken::hash(raw_key);

        // 4. Look up in cache (returns Arc, zero-clone)
        if let Some(key_obj) = lookup_cache(&state.key_cache, &hashed_token) {
            tracing::info!("Auth: Found key in cache, blocked={}", key_obj.blocked);
            if key_obj.blocked {
                return Err(AuthRejection::Blocked.into());
            }
            if key_obj.is_expired() {
                return Err(AuthRejection::Expired.into());
            }
            return Ok(RequireValidKey {
                key_object: key_obj,
                hashed_token,
            });
        }
        tracing::debug!("Auth: Key not found in cache");

        // 5. Cache miss: look up in DB (returns Arc)
        tracing::debug!("Auth: Looking up key in DB");
        let key_obj = lookup_db(state, &hashed_token)
            .await?
            .ok_or_else(|| {
                tracing::warn!("Auth: Key not found in DB, returning KeyNotFound");
                AuthRejection::KeyNotFound
            })?;

        tracing::info!("Auth: Found key in DB, blocked={}", key_obj.blocked);
        if key_obj.blocked {
            return Err(AuthRejection::Blocked.into());
        }
        if key_obj.is_expired() {
            return Err(AuthRejection::Expired.into());
        }

        // 6. Store in cache (Arc clone is cheap)
        state.key_cache.set(hashed_token, Arc::clone(&key_obj));

        Ok(RequireValidKey {
            key_object: key_obj,
            hashed_token,
        })
    }
}
