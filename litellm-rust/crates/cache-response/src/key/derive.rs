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

pub(crate) struct KeyContext<'a> {
    pub namespace: &'a str,
    pub scope: &'a CacheScope,
}

impl CacheKey {
    pub(crate) fn derive(input: &CacheKeyInput, context: &KeyContext<'_>) -> Self {
        let material = json!({
            "scope": match context.scope {
                CacheScope::Shared => None,
                CacheScope::Caller(caller) => Some(caller),
            },
            "input": input,
        });
        let hash = format!("{:x}", Sha256::digest(material.to_string()));
        Self::Native(match context.namespace {
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
