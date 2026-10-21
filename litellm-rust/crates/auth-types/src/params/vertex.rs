use serde::{Deserialize, Deserializer, Serialize, de::Unexpected};
use serde_json::Value;
use veil::Redact;

use super::ParamSpec;

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
    pub const CREDENTIALS: ParamSpec = ParamSpec {
        setting: "credentials",
        wire: &["vertex_credentials", "vertex_ai_credentials"],
        module_global: None,
        env: &["VERTEXAI_CREDENTIALS"],
    };
    pub const PROJECT: ParamSpec = ParamSpec {
        setting: "project",
        wire: &["vertex_project", "vertex_ai_project"],
        module_global: Some("vertex_project"),
        env: &["VERTEXAI_PROJECT"],
    };
    pub const LOCATION: ParamSpec = ParamSpec {
        setting: "location",
        wire: &["vertex_location", "vertex_ai_location"],
        module_global: Some("vertex_location"),
        env: &["VERTEXAI_LOCATION", "VERTEX_LOCATION"],
    };

    /// Every Vertex AI param Python reads, with both spellings, the module global
    /// `VertexBase.get_vertex_ai_*` consults, and the environment names it falls back to.
    pub const SPECS: [ParamSpec; 3] = [Self::CREDENTIALS, Self::PROJECT, Self::LOCATION];

    /// The wire names a host projects out of a caller's kwargs, derived from [`Self::SPECS`].
    pub fn fields() -> impl Iterator<Item = &'static str> {
        Self::SPECS
            .iter()
            .flat_map(|spec| spec.wire.iter().copied())
    }

    /// The value under a wire name, read through serde so the names can never drift from
    /// the struct.
    pub fn get(&self, wire: &str) -> Option<String> {
        serde_json::to_value(self)
            .ok()?
            .get(wire)?
            .as_str()
            .map(str::to_string)
    }

    /// The spec's value from these params, then the environment.
    pub fn resolve(
        &self,
        spec: &ParamSpec,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        spec.resolve(&|name| self.get(name), env_lookup)
    }

    pub fn credentials(&self) -> Option<String> {
        self.resolve(&Self::CREDENTIALS, &|_| None)
    }

    pub fn project(&self) -> Option<String> {
        self.resolve(&Self::PROJECT, &|_| None)
    }

    pub fn location(&self) -> Option<String> {
        self.resolve(&Self::LOCATION, &|_| None)
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
