use serde_json::json;
use sha2::{Digest, Sha256};

use super::CacheKeyInput;
use crate::CacheScope;

const KEY_VERSION: &str = "inference-v4";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheKey {
    Native(String),
    External(String),
}

impl CacheKey {
    pub(crate) fn derive(
        namespace: &str,
        surface: &str,
        scope: &CacheScope,
        input: &CacheKeyInput,
    ) -> Self {
        if let (CacheScope::Shared, CacheKeyInput::Preset(key)) = (scope, input) {
            return Self::Native(key.clone());
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
        Self::Native(match namespace {
            "" => format!("{KEY_VERSION}:{hash}"),
            namespace => format!("{namespace}:{KEY_VERSION}:{hash}"),
        })
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Native(key) | Self::External(key) => key,
        }
    }
}

impl From<CacheKey> for String {
    fn from(key: CacheKey) -> Self {
        match key {
            CacheKey::Native(key) | CacheKey::External(key) => key,
        }
    }
}
