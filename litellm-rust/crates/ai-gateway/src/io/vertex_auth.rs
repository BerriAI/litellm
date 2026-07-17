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

fn adc_cache_key() -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(b"adc");
    hasher.finalize().into()
}

fn content_cache_key(contents: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(b"inline:");
    hasher.update(contents.trim().as_bytes());
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
    let (key, built) = match credentials {
        None => (adc_cache_key(), None),
        Some(raw) => {
            let contents = load_credentials_contents(raw).await?;
            (content_cache_key(&contents), Some(contents))
        }
    };
    if let Some(existing) = credentials_cache().lock().await.get(&key) {
        return Ok(existing);
    }
    let credentials = match built {
        None => build_adc_credentials()?,
        Some(contents) => build_from_json(parse_credentials_json(&contents)?)?,
    };
    Ok(credentials_cache()
        .lock()
        .await
        .get_or_insert(key, credentials))
}

fn build_adc_credentials() -> CoreResult<AccessTokenCredentials> {
    AdcBuilder::default()
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .build_access_token_credentials()
        .map_err(|_| CoreError::Auth("Failed to load Vertex ADC credentials".to_string()))
}

async fn load_credentials_contents(raw: &str) -> CoreResult<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') {
        Ok(trimmed.to_string())
    } else {
        tokio::fs::read_to_string(trimmed)
            .await
            .map_err(|_| CoreError::Auth("Failed to read Vertex credentials file".to_string()))
    }
}

fn parse_credentials_json(contents: &str) -> CoreResult<Value> {
    serde_json::from_str(contents.trim())
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
    fn content_cache_key_distinguishes_adc_from_inline_and_matches_on_repeat() {
        let adc = adc_cache_key();
        let inline = content_cache_key("{\"type\":\"service_account\"}");
        assert_ne!(adc, inline);
        assert_eq!(
            inline,
            content_cache_key("  {\"type\":\"service_account\"}  ")
        );
    }

    #[test]
    fn content_cache_key_differs_for_distinct_credential_sources() {
        let adc = adc_cache_key();
        let first = content_cache_key("{\"type\":\"service_account\",\"client_email\":\"a\"}");
        let second = content_cache_key("{\"type\":\"service_account\",\"client_email\":\"b\"}");
        assert_ne!(first, second);
        assert_ne!(adc, first);
        assert_ne!(adc, second);
    }

    #[tokio::test]
    async fn file_backed_credentials_key_tracks_content_not_path() {
        let path = std::env::temp_dir().join(format!(
            "vertex-cred-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        tokio::fs::write(
            &path,
            b"{\"type\":\"service_account\",\"client_email\":\"old\"}",
        )
        .await
        .expect("writes first credential file");
        let first = load_credentials_contents(path.to_str().expect("utf-8 path"))
            .await
            .expect("reads first credential file");

        tokio::fs::write(
            &path,
            b"{\"type\":\"service_account\",\"client_email\":\"new\"}",
        )
        .await
        .expect("rotates credential file");
        let second = load_credentials_contents(path.to_str().expect("utf-8 path"))
            .await
            .expect("reads rotated credential file");
        tokio::fs::remove_file(&path).await.ok();

        assert_ne!(
            content_cache_key(&first),
            content_cache_key(&second),
            "rotating a credential file at the same path must not reuse the old cache entry"
        );
    }

    #[tokio::test]
    async fn build_from_json_builds_service_account_credentials() {
        build_from_json(json!({
            "type": "service_account",
            "project_id": "proj-1",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n",
            "client_email": "sa@proj-1.iam.gserviceaccount.com"
        }))
        .expect("service account dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_authorized_user_credentials() {
        build_from_json(json!({
            "type": "authorized_user",
            "client_id": "client-id.apps.googleusercontent.com",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token"
        }))
        .expect("authorized user dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_impersonated_service_account_credentials() {
        build_from_json(json!({
            "type": "impersonated_service_account",
            "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/target@proj-1.iam.gserviceaccount.com:generateAccessToken",
            "source_credentials": {
                "type": "authorized_user",
                "client_id": "client-id.apps.googleusercontent.com",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token"
            }
        }))
        .expect("impersonated service account dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_external_account_for_all_standard_source_mechanisms() {
        let base = |source: Value| {
            json!({
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/pr",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://sts.googleapis.com/v1/token",
                "credential_source": source
            })
        };

        build_from_json(base(json!({"file": "/var/run/secrets/token"})))
            .expect("file-sourced external account builds");
        build_from_json(base(json!({
            "url": "https://169.254.169.254/token",
            "headers": {"Metadata": "true"},
            "format": {"type": "json", "subject_token_field_name": "access_token"}
        })))
        .expect("url-sourced external account builds");
        build_from_json(base(json!({
            "executable": {"command": "/usr/bin/token-helper", "timeout_millis": 5000}
        })))
        .expect("executable-sourced external account builds");
        build_from_json(base(json!({
            "environment_id": "aws1",
            "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
            "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        })))
        .expect("aws-sourced external account builds");
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

    #[test]
    fn parse_credentials_json_reports_invalid_json_without_echoing_contents() {
        let err = parse_credentials_json("{not-valid-json-secret-value")
            .expect_err("invalid json rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(
                    !message.contains("not-valid-json-secret-value"),
                    "{message}"
                );
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_credentials_contents_missing_file_error_omits_attacker_controlled_path() {
        let err = load_credentials_contents("/attacker/controlled/secret-credentials-path.json")
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
