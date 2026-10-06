use serde_json::json;
use sha2::{Digest, Sha256};

use super::CacheKeyInput;
use crate::CacheScope;

const KEY_VERSION: &str = "inference-v4";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheKey(String);

impl CacheKey {
    pub fn delegated(key: String) -> Self {
        Self(key)
    }

    pub(crate) fn derive(
        namespace: &str,
        surface: &str,
        scope: &CacheScope,
        input: &CacheKeyInput,
    ) -> Self {
        if let (CacheScope::Shared, CacheKeyInput::Preset(key)) = (scope, input) {
            return Self(key.clone());
        }
        let material = json!({
            "surface": surface,
            "scope": match scope {
                CacheScope::Shared => None,
                CacheScope::Isolated(scope) => Some(scope),
            },
            "input": input,
        });
        let hash = format!("{:x}", Sha256::digest(material.to_string()));
        Self(match namespace {
            "" => format!("{KEY_VERSION}:{hash}"),
            namespace => format!("{namespace}:{KEY_VERSION}:{hash}"),
        })
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<CacheKey> for String {
    fn from(key: CacheKey) -> Self {
        key.0
    }
}
