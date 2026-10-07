use serde_json::json;
use sha2::{Digest, Sha256};

use super::{CacheCredential, CacheKeyInput, CacheScope, target::CacheTarget};

const KEY_VERSION: &str = "v0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheKey {
    Derived(String),
    Supplied(String),
}

impl CacheKey {
    pub(crate) fn derive(input: &CacheKeyInput, scope: &CacheScope) -> Self {
        let material = json!({
            "credential": scope.credential.as_ref().map(CacheCredential::as_str),
            "surface": input.surface,
            "target": CacheTarget::resolve(scope, &input.deployment),
            "parameters": input.parameters,
        });
        let hash = format!("{:x}", Sha256::digest(material.to_string()));
        Self::Derived(format!("{KEY_VERSION}:{hash}"))
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Derived(key) | Self::Supplied(key) => key,
        }
    }
}

impl From<CacheKey> for String {
    fn from(key: CacheKey) -> Self {
        match key {
            CacheKey::Derived(key) | CacheKey::Supplied(key) => key,
        }
    }
}
