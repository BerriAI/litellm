use crate::SecretValue;

#[derive(Clone, PartialEq, Eq, veil::Redact)]
pub enum Secret {
    String(SecretValue),
    Bool(#[redact] bool),
    Json(#[redact] serde_json::Value),
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
            serde_json::Value::Bool(value) => Self::Bool(value),
            value => Self::Json(value),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.expose()),
            Self::Bool(_) | Self::Json(_) => None,
        }
    }
}
