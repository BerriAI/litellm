use std::time::Duration;

use litellm_python_compat::Value;

use crate::{DecodeError, FromValue, IntoValue, Number};

#[derive(Clone, Debug, PartialEq)]
pub struct Cooldown {
    pub exception_received: String,
    pub status_code: String,
    pub timestamp: Number,
    pub cooldown_time: Number,
}

impl Cooldown {
    pub(super) fn remaining(&self, now: Duration) -> f64 {
        self.timestamp.as_f64() + self.cooldown_time.as_f64() - now.as_secs_f64()
    }
}

/// The `CooldownCacheValue` dict the router reads and writes.
impl IntoValue for Cooldown {
    fn into_value(self) -> Value {
        let field = |name: &str, value: Value| (Value::Str(name.to_owned()), value);
        Value::Dict(vec![
            field("exception_received", Value::Str(self.exception_received)),
            field("status_code", Value::Str(self.status_code)),
            field("timestamp", self.timestamp.to_value()),
            field("cooldown_time", self.cooldown_time.to_value()),
        ])
    }
}

impl FromValue for Cooldown {
    fn from_value(value: &Value) -> Result<Self, DecodeError> {
        let Value::Dict(fields) = value else {
            return Err(DecodeError::Shape("a cooldown is a dict"));
        };
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
                .map(|(_, value)| value)
        };
        let text = |name: &'static str| match field(name) {
            Some(Value::Str(text)) => Ok(text.clone()),
            _ => Err(DecodeError::Shape(name)),
        };
        let number = |name: &'static str| {
            field(name)
                .and_then(Number::from_value)
                .ok_or(DecodeError::Shape(name))
        };
        Ok(Self {
            exception_received: text("exception_received")?,
            status_code: text("status_code")?,
            timestamp: number("timestamp")?,
            cooldown_time: number("cooldown_time")?,
        })
    }
}
