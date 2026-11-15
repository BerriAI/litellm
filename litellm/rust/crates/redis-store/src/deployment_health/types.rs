use litellm_python_compat::{Value, truthy::truthy};

use crate::{DecodeError, FromValue, IntoValue, Number};

/// One background health check's result, the `DeploymentHealthStateValue` dict the proxy writes.
#[derive(Clone, Debug, PartialEq)]
pub struct HealthState {
    pub is_healthy: bool,
    pub timestamp: Number,
    pub reason: String,
}

/// Every deployment's last health check by id, in the order the writer merged them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HealthStates(pub Vec<(String, HealthState)>);

impl HealthState {
    /// What main's readers take from a record: extra fields are ignored, any falsy flag is
    /// unhealthy and a missing one healthy, and a timestamp that is no number reads as `0` (stale).
    fn read(fields: &[(Value, Value)]) -> Self {
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
                .map(|(_, value)| value)
        };
        Self {
            is_healthy: field("is_healthy").is_none_or(truthy),
            timestamp: field("timestamp")
                .and_then(Number::from_value)
                .unwrap_or(Number::Float(0.0)),
            reason: match field("reason") {
                Some(Value::Str(reason)) => reason.clone(),
                _ => String::new(),
            },
        }
    }
}

impl IntoValue for HealthState {
    fn into_value(self) -> Value {
        let field = |name: &str, value: Value| (Value::Str(name.to_owned()), value);
        Value::Dict(vec![
            field("is_healthy", Value::Bool(self.is_healthy)),
            field("timestamp", self.timestamp.to_value()),
            field("reason", Value::Str(self.reason)),
        ])
    }
}

/// A record that is no dict, under a key that is no string, is dropped alone.
impl FromValue for HealthStates {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        let Value::Dict(records) = value else {
            return Err(DecodeError::Shape("deployment health states are a dict"));
        };
        Ok(Self(
            records
                .iter()
                .filter_map(|record| match record {
                    (Value::Str(id), Value::Dict(fields)) => {
                        Some((id.clone(), HealthState::read(fields)))
                    }
                    _ => None,
                })
                .collect(),
        ))
    }
}

impl IntoValue for HealthStates {
    fn into_value(self) -> Value {
        Value::Dict(
            self.0
                .into_iter()
                .map(|(id, state)| (Value::Str(id), state.into_value()))
                .collect(),
        )
    }
}
