use std::time::Duration;

use litellm_python_compat::{Value, json};

use crate::Number;

/// One message on the invalidation channel: every worker drops its memory copy of a key, or writes
/// the value a spend reset just committed so its own echo cannot erase it.
#[derive(Clone, Debug, PartialEq)]
pub enum Invalidation {
    Evict(String),
    Set {
        cache_key: String,
        value: Number,
        ttl: Option<Number>,
    },
}

impl Invalidation {
    pub fn cache_key(&self) -> &str {
        match self {
            Self::Evict(cache_key) | Self::Set { cache_key, .. } => cache_key,
        }
    }

    /// The memory TTL a `Set` carries, `None` when it names none or one a `Duration` cannot hold.
    pub(super) fn memory_ttl(ttl: Option<Number>) -> Option<Duration> {
        ttl.and_then(|seconds| Duration::try_from_secs_f64(seconds.as_f64()).ok())
    }

    /// `_cache_invalidation_message_json`: `json.dumps` of the fields that are not `None`.
    pub fn encode(&self) -> String {
        let field = |name: &str, value: Value| (Value::from(name), value);
        let fields = match self {
            Self::Evict(cache_key) => vec![field("cache_key", Value::from(cache_key.as_str()))],
            Self::Set {
                cache_key,
                value,
                ttl,
            } => [
                Some(field("cache_key", Value::from(cache_key.as_str()))),
                Some(field("new_value", value.to_value())),
                ttl.map(|ttl| field("ttl", ttl.to_value())),
            ]
            .into_iter()
            .flatten()
            .collect(),
        };
        json::dumps(&Value::Dict(fields)).expect("json.dumps never refuses a str and numbers")
    }

    /// `_message_from_data`: anything but a JSON object with a string `cache_key` is ignored, a
    /// `new_value` or `ttl` that is not a number reads as absent, and a number reads as `float()`.
    pub fn decode(payload: &[u8]) -> Option<Self> {
        let Value::Dict(fields) = json::loads(&String::from_utf8_lossy(payload)).ok()? else {
            return None;
        };
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
                .map(|(_, value)| value)
        };
        let number = |name: &str| {
            field(name)
                .and_then(Number::from_value)
                .map(|number| Number::Float(number.as_f64()))
        };
        let Some(Value::Str(cache_key)) = field("cache_key") else {
            return None;
        };
        Some(match number("new_value") {
            Some(value) => Self::Set {
                cache_key: cache_key.clone(),
                value,
                ttl: number("ttl"),
            },
            None => Self::Evict(cache_key.clone()),
        })
    }
}
