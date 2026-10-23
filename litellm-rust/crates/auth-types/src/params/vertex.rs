use litellm_layered::{Layered, read};
use serde::{Deserialize, Deserializer, Serialize, de::Unexpected};
use serde_json::Value;
use veil::Redact;

#[derive(Layered)]
pub struct VertexKeys {
    #[layer(
        litellm_param = ["vertex_credentials", "vertex_ai_credentials"],
        deserialize_with = optional_credential_text,
        env = "VERTEXAI_CREDENTIALS",
        read = read::text,
        secret
    )]
    pub credentials: String,
    #[layer(
        litellm_param = ["vertex_project", "vertex_ai_project"],
        litellm_setting = "vertex_project",
        env = "VERTEXAI_PROJECT",
        read = read::text
    )]
    pub project: String,
    #[layer(
        litellm_param = ["vertex_location", "vertex_ai_location"],
        litellm_setting = "vertex_location",
        env = ["VERTEXAI_LOCATION", "VERTEX_LOCATION"],
        read = read::text
    )]
    pub location: String,
}

/// The Vertex AI fields of Python's `GenericLiteLLMParams`, both the current names and the
/// `vertex_ai_*` spellings `VertexBase.safe_get_vertex_ai_*` still read.
#[derive(Redact, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VertexParams {
    #[serde(default, deserialize_with = "optional_credential_text")]
    #[redact(with = "[REDACTED]")]
    pub vertex_credentials: Option<String>,
    #[serde(default)]
    pub vertex_project: Option<String>,
    #[serde(default)]
    pub vertex_location: Option<String>,
    #[serde(default, deserialize_with = "optional_credential_text")]
    #[redact(with = "[REDACTED]")]
    pub vertex_ai_credentials: Option<String>,
    #[serde(default)]
    pub vertex_ai_project: Option<String>,
    #[serde(default)]
    pub vertex_ai_location: Option<String>,
}

impl VertexParams {
    pub fn keys(&self) -> VertexKeysPartial {
        VertexKeysPartial::from_litellm_params(&super::as_litellm_params(self))
    }

    pub fn credentials(&self) -> Option<String> {
        self.keys().credentials
    }

    pub fn project(&self) -> Option<String> {
        self.keys().project
    }

    pub fn location(&self) -> Option<String> {
        self.keys().location
    }
}

/// A credential is a service account JSON text or a path to one; a YAML or kwargs mapping is
/// accepted as the JSON text it spells, the way Python passes a dict through.
fn optional_credential_text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(Value::Object(object)) if object.is_empty() => Ok(None),
        Some(Value::Object(object)) => serde_json::to_string(&object)
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(other) => Err(serde::de::Error::invalid_type(
            Unexpected::Other(value_kind(&other)),
            &"a string or an object",
        )),
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}
