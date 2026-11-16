use serde::Deserialize;

use crate::AdditionalFields;

/// A config value YAML may give as one string or a list of them.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Box<[String]>),
}

impl OneOrMany {
    pub fn values(&self) -> &[String] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
pub enum RoutingPath {
    #[default]
    #[serde(rename = "oauth2")]
    OAuth2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JwtRoutingOverride {
    pub iss: OneOrMany,
    #[serde(default)]
    pub client_id: Option<OneOrMany>,
    #[serde(default)]
    pub scope: Option<OneOrMany>,
    #[serde(default)]
    pub aud: Option<OneOrMany>,
    #[serde(default)]
    pub path: RoutingPath,
}

/// `general_settings.litellm_jwtauth`, with `LiteLLM_JWTAuth`'s defaults. `custom_validate` is
/// the operator's import path: only the host can run it, so the core only learns that it is set.
#[derive(Clone, Default, Deserialize, veil::Redact)]
#[serde(default)]
pub struct JwtAuthSettings {
    pub custom_validate: Option<String>,
    pub routing_overrides: Option<Box<[JwtRoutingOverride]>>,
    #[serde(flatten)]
    #[redact(with = "[REDACTED]")]
    pub additional_fields: AdditionalFields,
}
