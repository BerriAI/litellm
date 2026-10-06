use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::CacheScope;

const KEY_VERSION: &str = "inference-v3";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTarget {
    ModelGroup(String),
    Model(String),
}

impl CacheTarget {
    pub fn resolve(model: &str, model_group: Option<&str>) -> Self {
        match model_group {
            Some(group) => Self::ModelGroup(group.to_owned()),
            None => Self::Model(model.to_owned()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKeyInput {
    Preset(String),
    Request {
        target: CacheTarget,
        parameters: Value,
    },
}

impl CacheKeyInput {
    pub fn request(target: CacheTarget, parameters: Value) -> Self {
        let parameters = match canonical(parameters) {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .filter(|(_, value)| !value.is_null())
                    .collect(),
            ),
            value => value,
        };
        Self::Request { target, parameters }
    }
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(name, value)| (name, canonical(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        value => value,
    }
}

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
        let material = serde_json::json!({
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
