use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize)]
pub struct CacheKeyField {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheKeyTransport {
    pub provider: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheKeyRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct CacheKeyInput {
    pub fields: Vec<CacheKeyField>,
    pub preset: Option<String>,
    pub namespace: Option<String>,
    pub transport: Option<CacheKeyTransport>,
    pub rewritten_request: Option<CacheKeyRequest>,
}

impl CacheKeyInput {
    pub fn from_parameters(parameters: Value) -> Self {
        let fields = match canonical(parameters) {
            Value::Object(fields) => fields
                .into_iter()
                .map(|(name, value)| CacheKeyField {
                    name,
                    value: (!value.is_null()).then(|| value.to_string()),
                })
                .collect(),
            value => vec![CacheKeyField {
                name: "request".into(),
                value: Some(value.to_string()),
            }],
        };
        Self {
            fields,
            ..Self::default()
        }
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

    pub(crate) fn derive(input: &CacheKeyInput) -> Self {
        Self(get_cache_key(input))
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

fn get_cache_key(input: &CacheKeyInput) -> String {
    if let Some(preset) = &input.preset {
        return preset.clone();
    }
    let mut digest = Sha256::new();
    for field in &input.fields {
        if let Some(value) = &field.value {
            digest.update(field.name.as_bytes());
            digest.update(b": ");
            digest.update(value.as_bytes());
        }
    }
    if let Some(transport) = &input.transport {
        digest.update(b"transport: ");
        digest.update(serde_json::json!(transport).to_string().as_bytes());
    }
    if let Some(request) = &input.rewritten_request {
        digest.update(b"wire_changes: ");
        digest.update(serde_json::json!(request).to_string().as_bytes());
    }
    let hash = format!("{:x}", digest.finalize());
    input
        .namespace
        .as_deref()
        .filter(|namespace| !namespace.is_empty())
        .map_or(hash.clone(), |namespace| format!("{namespace}:{hash}"))
}
