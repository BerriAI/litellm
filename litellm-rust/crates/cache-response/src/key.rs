use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKeyParticipation {
    Always,
    ProviderOptIn,
    Never,
}

impl CacheKeyParticipation {
    pub fn includes(self, include_provider_parameters: bool) -> bool {
        match self {
            Self::Always => true,
            Self::ProviderOptIn => include_provider_parameters,
            Self::Never => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CacheKeyField {
    pub name: String,
    pub value: Option<String>,
    pub participation: CacheKeyParticipation,
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
    pub include_provider_parameters: bool,
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
                    participation: CacheKeyParticipation::Always,
                })
                .collect(),
            value => vec![CacheKeyField {
                name: "request".into(),
                value: Some(value.to_string()),
                participation: CacheKeyParticipation::Always,
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

#[derive(Clone, Debug, Default)]
pub struct CacheKeyContext {
    pub model_group: Option<String>,
    pub caching_groups: Vec<(Vec<String>, String)>,
    pub file_checksum: Option<String>,
    pub file_object_name: Option<String>,
    pub metadata_file_name: Option<String>,
    pub parameters_file_name: Option<String>,
}

impl CacheKeyContext {
    pub fn project(&self, input: CacheKeyInput) -> CacheKeyInput {
        let group = self
            .model_group
            .as_ref()
            .filter(|value| !value.is_empty())
            .and_then(|model| {
                self.caching_groups
                    .iter()
                    .find(|(models, _)| models.contains(model))
            });

        CacheKeyInput {
            fields: input
                .fields
                .into_iter()
                .map(|field| {
                    let value = match field.name.as_str() {
                        "model" => group
                            .map(|(_, formatted)| formatted.clone())
                            .or_else(|| self.model_group.clone().filter(|value| !value.is_empty()))
                            .or(field.value),
                        "file" => [
                            &self.file_checksum,
                            &self.file_object_name,
                            &self.metadata_file_name,
                            &self.parameters_file_name,
                        ]
                        .into_iter()
                        .flatten()
                        .find(|value| !value.is_empty())
                        .cloned(),
                        _ => field.value,
                    };
                    CacheKeyField { value, ..field }
                })
                .collect(),
            ..input
        }
    }
}

pub fn get_cache_key(input: &CacheKeyInput) -> String {
    if let Some(preset) = &input.preset {
        return preset.clone();
    }
    let mut digest = Sha256::new();
    for field in &input.fields {
        if field
            .participation
            .includes(input.include_provider_parameters)
            && let Some(value) = &field.value
        {
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
