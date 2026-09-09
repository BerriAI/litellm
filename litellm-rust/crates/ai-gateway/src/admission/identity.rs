//! Who is calling: the master key, or a virtual key resolved through the Python proxy.
//!
//! Virtual keys are cached by their SHA-256 hash for the life of the process, so the network
//! round trip to `/key/info` happens once per key, not once per request.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::auth::hash_token;
use crate::constants::{KEY_INFO_TIMEOUT_SECS, PROXY_KEY_INFO_PATH};

/// Limits attached to a virtual key. `None` means unlimited, as in the proxy.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct KeyLimits {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub spend: f64,
    #[serde(default)]
    pub tpm_limit: Option<u64>,
    #[serde(default)]
    pub rpm_limit: Option<u64>,
}

impl KeyLimits {
    pub fn allows_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|allowed| allowed == model)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Identity {
    Master,
    VirtualKey {
        key_hash: String,
        limits: Arc<KeyLimits>,
    },
}

impl Identity {
    pub fn key_hash(&self) -> &str {
        match self {
            Identity::Master => "litellm_proxy_master_key",
            Identity::VirtualKey { key_hash, .. } => key_hash,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum IdentityError {
    #[error("missing or invalid bearer token")]
    Unauthorized,
    #[error("key lookup failed: {0}")]
    LookupFailed(String),
}

#[derive(Deserialize)]
struct KeyInfoResponse {
    info: KeyLimits,
}

/// Resolves bearer tokens; misses go to the proxy's `/key/info`.
pub struct IdentityCache {
    master_key: Option<Arc<str>>,
    proxy_base_url: String,
    http: reqwest::Client,
    cache: RwLock<HashMap<String, Arc<KeyLimits>>>,
}

impl IdentityCache {
    pub fn new(master_key: Option<Arc<str>>, proxy_base_url: String) -> Self {
        Self {
            master_key,
            proxy_base_url: proxy_base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(KEY_INFO_TIMEOUT_SECS))
                .build()
                .unwrap_or_default(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Seed the cache, so tests and offline hosts never call the proxy.
    pub fn insert(&self, token: &str, limits: KeyLimits) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(hash_token(token), Arc::new(limits));
        }
    }

    pub async fn resolve(&self, token: Option<&str>) -> Result<Identity, IdentityError> {
        let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) else {
            return Err(IdentityError::Unauthorized);
        };
        if let Some(master) = self.master_key.as_deref()
            && bool::from(token.as_bytes().ct_eq(master.as_bytes()))
        {
            return Ok(Identity::Master);
        }
        let key_hash = hash_token(token);
        let cached = self
            .cache
            .read()
            .ok()
            .and_then(|cache| cache.get(&key_hash).cloned());
        let limits = match cached {
            Some(limits) => limits,
            None => {
                let limits = Arc::new(self.fetch(token).await?);
                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(key_hash.clone(), Arc::clone(&limits));
                }
                limits
            }
        };
        Ok(Identity::VirtualKey { key_hash, limits })
    }

    async fn fetch(&self, token: &str) -> Result<KeyLimits, IdentityError> {
        let Some(master) = self.master_key.as_deref() else {
            return Err(IdentityError::Unauthorized);
        };
        let response = self
            .http
            .get(format!("{}{PROXY_KEY_INFO_PATH}", self.proxy_base_url))
            .query(&[("key", token)])
            .bearer_auth(master)
            .send()
            .await
            .map_err(|error| IdentityError::LookupFailed(error.without_url().to_string()))?;
        match response.status().as_u16() {
            200 => response
                .json::<KeyInfoResponse>()
                .await
                .map(|body| body.info)
                .map_err(|error| IdentityError::LookupFailed(error.without_url().to_string())),
            400..=404 => Err(IdentityError::Unauthorized),
            status => Err(IdentityError::LookupFailed(format!(
                "proxy answered {status}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> IdentityCache {
        IdentityCache::new(
            Some(Arc::from("sk-master")),
            "http://127.0.0.1:1".to_string(),
        )
    }

    #[tokio::test]
    async fn master_key_is_unlimited_and_never_looked_up() {
        assert_eq!(
            cache().resolve(Some("sk-master")).await,
            Ok(Identity::Master)
        );
    }

    #[tokio::test]
    async fn missing_token_is_unauthorized() {
        assert_eq!(
            cache().resolve(None).await,
            Err(IdentityError::Unauthorized)
        );
        assert_eq!(
            cache().resolve(Some("  ")).await,
            Err(IdentityError::Unauthorized)
        );
    }

    #[tokio::test]
    async fn seeded_virtual_key_resolves_from_cache_without_network() {
        let cache = cache();
        let limits = KeyLimits {
            models: vec!["claude".to_string()],
            max_budget: Some(10.0),
            spend: 1.5,
            tpm_limit: Some(1000),
            rpm_limit: None,
        };
        cache.insert("sk-virtual", limits.clone());
        let identity = cache.resolve(Some("sk-virtual")).await.unwrap();
        match identity {
            Identity::VirtualKey {
                key_hash,
                limits: resolved,
            } => {
                assert_eq!(key_hash, hash_token("sk-virtual"));
                assert_eq!(*resolved, limits);
            }
            Identity::Master => panic!("virtual key resolved as master"),
        }
    }

    #[tokio::test]
    async fn unknown_key_lookup_failure_is_reported_not_admitted() {
        let error = cache().resolve(Some("sk-unknown")).await.unwrap_err();
        assert!(matches!(error, IdentityError::LookupFailed(_)), "{error:?}");
    }

    #[test]
    fn empty_model_list_allows_every_model() {
        assert!(KeyLimits::default().allows_model("anything"));
        let limits = KeyLimits {
            models: vec!["a".to_string()],
            ..KeyLimits::default()
        };
        assert!(limits.allows_model("a"));
        assert!(!limits.allows_model("b"));
    }
}
