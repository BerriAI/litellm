use std::collections::HashMap;
use std::sync::{Once, OnceLock};

use google_cloud_auth::credentials::service_account::AccessSpecifier;
use google_cloud_auth::credentials::{
    external_account, impersonated, service_account, user_account, AccessTokenCredentials,
    Builder as AdcBuilder,
};
use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::constants::CLOUD_PLATFORM_SCOPE;

type CacheKey = [u8; 32];

fn ensure_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn credentials_cache() -> &'static Mutex<HashMap<CacheKey, AccessTokenCredentials>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, AccessTokenCredentials>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(credentials: Option<&str>) -> CacheKey {
    let mut hasher = Sha256::new();
    match credentials {
        None => hasher.update(b"adc"),
        Some(raw) => {
            hasher.update(b"inline:");
            hasher.update(raw.trim().as_bytes());
        }
    }
    hasher.finalize().into()
}

pub async fn mint_vertex_bearer(credentials: Option<&str>) -> CoreResult<String> {
    ensure_crypto_provider();
    let token = resolve_credentials(credentials)
        .await?
        .access_token()
        .await
        .map_err(|err| CoreError::Auth(format!("Failed to obtain Vertex access token: {err}")))?;
    Ok(token.token)
}

async fn resolve_credentials(credentials: Option<&str>) -> CoreResult<AccessTokenCredentials> {
    let key = cache_key(credentials);
    if let Some(existing) = credentials_cache().lock().await.get(&key) {
        return Ok(existing.clone());
    }
    let built = build_credentials(credentials).await?;
    let mut cache = credentials_cache().lock().await;
    Ok(cache.entry(key).or_insert(built).clone())
}

async fn build_credentials(credentials: Option<&str>) -> CoreResult<AccessTokenCredentials> {
    match credentials {
        None => AdcBuilder::default()
            .with_scopes([CLOUD_PLATFORM_SCOPE])
            .build_access_token_credentials()
            .map_err(|err| {
                CoreError::Auth(format!("Failed to load Vertex ADC credentials: {err}"))
            }),
        Some(raw) => build_from_json(load_credentials_json(raw).await?),
    }
}

async fn load_credentials_json(raw: &str) -> CoreResult<Value> {
    let trimmed = raw.trim();
    let contents = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else {
        tokio::fs::read_to_string(trimmed).await.map_err(|err| {
            CoreError::Auth(format!("Failed to read Vertex credentials file: {err}"))
        })?
    };
    serde_json::from_str(&contents)
        .map_err(|_| CoreError::Auth("Vertex credentials are not valid JSON".to_string()))
}

fn build_from_json(json: Value) -> CoreResult<AccessTokenCredentials> {
    let scopes = [CLOUD_PLATFORM_SCOPE];
    let credentials = match json.get("type").and_then(Value::as_str) {
        Some("service_account") => service_account::Builder::new(json)
            .with_access_specifier(AccessSpecifier::from_scopes(scopes))
            .build_access_token_credentials(),
        Some("authorized_user") => user_account::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some("external_account") => external_account::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some("impersonated_service_account") => impersonated::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some(other) => {
            return Err(CoreError::Auth(format!(
                "Unsupported Vertex credential type: {other}"
            )))
        }
        None => {
            return Err(CoreError::Auth(
                "Vertex credentials JSON is missing the required `type` field".to_string(),
            ))
        }
    };
    credentials.map_err(|err| CoreError::Auth(format!("Failed to load Vertex credentials: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cache_key_distinguishes_adc_from_inline_and_matches_on_repeat() {
        let adc = cache_key(None);
        let inline = cache_key(Some("{\"type\":\"service_account\"}"));
        assert_ne!(adc, inline);
        assert_eq!(
            inline,
            cache_key(Some("  {\"type\":\"service_account\"}  "))
        );
    }

    #[test]
    fn cache_key_differs_for_distinct_credential_sources() {
        let adc = cache_key(None);
        let first = cache_key(Some(
            "{\"type\":\"service_account\",\"client_email\":\"a\"}",
        ));
        let second = cache_key(Some(
            "{\"type\":\"service_account\",\"client_email\":\"b\"}",
        ));
        assert_ne!(first, second);
        assert_ne!(adc, first);
        assert_ne!(adc, second);
    }

    #[test]
    fn build_from_json_rejects_unknown_credential_type() {
        let err =
            build_from_json(json!({"type": "totally_made_up"})).expect_err("unknown type rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn build_from_json_requires_type_field() {
        let err = build_from_json(json!({"client_email": "x"})).expect_err("missing type rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[tokio::test]
    async fn load_credentials_json_reports_invalid_json_without_echoing_contents() {
        let err = load_credentials_json("not-json-and-not-a-path{")
            .await
            .expect_err("invalid json rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(!message.contains("not-json-and-not-a-path"), "{message}");
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }
}
