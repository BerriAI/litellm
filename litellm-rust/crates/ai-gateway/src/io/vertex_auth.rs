use std::collections::{HashMap, VecDeque};
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

use crate::constants::{CLOUD_PLATFORM_SCOPE, VERTEX_CREDENTIALS_CACHE_CAPACITY};

type CacheKey = [u8; 32];

struct BoundedCache<V> {
    capacity: usize,
    entries: HashMap<CacheKey, V>,
    order: VecDeque<CacheKey>,
}

impl<V: Clone> BoundedCache<V> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, key: &CacheKey) -> Option<V> {
        self.entries.get(key).cloned()
    }

    fn get_or_insert(&mut self, key: CacheKey, value: V) -> V {
        if let Some(existing) = self.entries.get(&key) {
            return existing.clone();
        }
        while self.entries.len() >= self.capacity {
            match self.order.pop_front() {
                Some(evicted) => {
                    self.entries.remove(&evicted);
                }
                None => break,
            }
        }
        self.order.push_back(key);
        self.entries.insert(key, value.clone());
        value
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn ensure_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn credentials_cache() -> &'static Mutex<BoundedCache<AccessTokenCredentials>> {
    static CACHE: OnceLock<Mutex<BoundedCache<AccessTokenCredentials>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BoundedCache::new(VERTEX_CREDENTIALS_CACHE_CAPACITY)))
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
        .map_err(|_| CoreError::Auth("Failed to obtain Vertex access token".to_string()))?;
    Ok(token.token)
}

async fn resolve_credentials(credentials: Option<&str>) -> CoreResult<AccessTokenCredentials> {
    let key = cache_key(credentials);
    if let Some(existing) = credentials_cache().lock().await.get(&key) {
        return Ok(existing);
    }
    let built = build_credentials(credentials).await?;
    Ok(credentials_cache().lock().await.get_or_insert(key, built))
}

async fn build_credentials(credentials: Option<&str>) -> CoreResult<AccessTokenCredentials> {
    match credentials {
        None => AdcBuilder::default()
            .with_scopes([CLOUD_PLATFORM_SCOPE])
            .build_access_token_credentials()
            .map_err(|_| CoreError::Auth("Failed to load Vertex ADC credentials".to_string())),
        Some(raw) => build_from_json(load_credentials_json(raw).await?),
    }
}

async fn load_credentials_json(raw: &str) -> CoreResult<Value> {
    let trimmed = raw.trim();
    let contents = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else {
        tokio::fs::read_to_string(trimmed)
            .await
            .map_err(|_| CoreError::Auth("Failed to read Vertex credentials file".to_string()))?
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
        Some(_) => {
            return Err(CoreError::Auth(
                "Unsupported Vertex credential type".to_string(),
            ))
        }
        None => {
            return Err(CoreError::Auth(
                "Vertex credentials JSON is missing the required `type` field".to_string(),
            ))
        }
    };
    credentials.map_err(|_| CoreError::Auth("Failed to load Vertex credentials".to_string()))
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

    #[test]
    fn build_from_json_unknown_type_error_omits_attacker_controlled_type() {
        let err = build_from_json(json!({"type": "attacker-controlled-type-string"}))
            .expect_err("unknown type rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(
                    !message.contains("attacker-controlled-type-string"),
                    "{message}"
                );
            }
            other => panic!("expected auth error, got {other:?}"),
        }
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

    #[tokio::test]
    async fn load_credentials_json_missing_file_error_omits_attacker_controlled_path() {
        let err = load_credentials_json("/attacker/controlled/secret-credentials-path.json")
            .await
            .expect_err("missing file rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(!message.contains("attacker"), "{message}");
                assert!(!message.contains("secret-credentials-path"), "{message}");
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    fn seq_key(n: u32) -> CacheKey {
        let mut key = [0_u8; 32];
        key[..4].copy_from_slice(&n.to_le_bytes());
        key
    }

    #[test]
    fn bounded_cache_evicts_oldest_and_never_exceeds_capacity() {
        let mut cache: BoundedCache<u32> = BoundedCache::new(4);
        for n in 0..100_u32 {
            cache.get_or_insert(seq_key(n), n);
            assert!(cache.len() <= 4, "size stayed bounded at {}", cache.len());
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.get(&seq_key(99)), Some(99));
        assert_eq!(cache.get(&seq_key(0)), None);
        assert_eq!(cache.get(&seq_key(95)), None);
    }

    #[test]
    fn bounded_cache_reuses_existing_entry_without_replacing_or_growing() {
        let mut cache: BoundedCache<u32> = BoundedCache::new(4);
        assert_eq!(cache.get_or_insert(seq_key(7), 1), 1);
        assert_eq!(cache.get_or_insert(seq_key(7), 2), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&seq_key(7)), Some(1));
    }
}
