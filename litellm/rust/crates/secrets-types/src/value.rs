use litellm_secret_value::{Redacted, SecretValue};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Secret {
    String(SecretValue),
    Bool(Redacted<bool>),
    Json(Redacted<serde_json::Value>),
}

#[derive(Debug)]
pub enum PythonSecretRead {
    Value(Option<Secret>),
    PrimaryJson(SecretValue),
}

impl From<SecretValue> for Secret {
    fn from(value: SecretValue) -> Self {
        Self::String(value)
    }
}

impl Secret {
    pub fn from_json(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(value) => Self::String(SecretValue::new(value)),
            serde_json::Value::Bool(value) => Self::Bool(Redacted::new(value)),
            value => Self::Json(Redacted::new(value)),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.expose()),
            Self::Bool(_) | Self::Json(_) => None,
        }
    }
}
