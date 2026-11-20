use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, de::Error as _};
use veil::Redact;

use crate::{Object, Value};

/// A Redis section as `config.yaml` spells it, before any surface reads it.
#[derive(Clone, Default, Deserialize, Redact)]
#[serde(transparent)]
pub struct RedisSettings {
    #[redact(with = "[REDACTED]")]
    fields: BTreeMap<String, Value>,
}

impl RedisSettings {
    pub fn fields(&self) -> &BTreeMap<String, Value> {
        &self.fields
    }
}

#[derive(Clone, Debug)]
pub enum CacheSettings {
    Redis(Box<RedisSettings>),
    Other(Object),
}

impl<'de> Deserialize<'de> for CacheSettings {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let is_redis = matches!(fields.get("type"), Some(Value::String(kind)) if kind == "redis");
        if !is_redis {
            return Ok(Self::Other(Object::from_fields(fields)));
        }
        let mapping = fields
            .into_iter()
            .map(|(key, value)| (Value::String(key), value))
            .collect();
        serde_yaml_ng::from_value(Value::Mapping(mapping))
            .map(|settings| Self::Redis(Box::new(settings)))
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::RedisSettings;
    use rstest::rstest;

    #[rstest]
    #[case::scalar(serde_json::json!("credential-value"))]
    #[case::object(serde_json::json!({"secret": "credential-value"}))]
    #[case::array(serde_json::json!(["credential-value"]))]
    fn credentials_are_redacted(#[case] password: serde_json::Value) {
        let settings: RedisSettings = serde_json::from_value(serde_json::json!({
            "host": "redis", "password": password,
        }))
        .unwrap();
        let debug = format!("{settings:?}");
        assert!(!debug.contains("credential-value"));
        assert!(debug.contains("[REDACTED]"));
    }
}
