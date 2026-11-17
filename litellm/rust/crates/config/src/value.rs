use std::{collections::BTreeMap, fmt, ops::Deref};

use litellm_python_compat::{Value as PythonValue, flags::lax_bool, number::float};
use litellm_router_types::Spelled;
use serde::{Deserialize, Deserializer};

pub type Value = serde_yaml_ng::Value;
/// The keys a config section has beyond its typed fields, printed as keys only.
pub type AdditionalFields = Object;

#[derive(Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct Object(BTreeMap<String, Value>);

impl Object {
    pub(crate) fn from_fields(fields: BTreeMap<String, Value>) -> Self {
        Self(fields)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Deref for Object {
    type Target = BTreeMap<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Object")
            .field("keys", &self.0.keys())
            .finish()
    }
}

/// Python's callback shorthand: one entry or a list of them, kept as the values written.
pub(crate) fn one_or_many<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<Value>>, D::Error> {
    Ok(
        Option::<Value>::deserialize(deserializer)?.map(|value| match value {
            Value::Sequence(values) => values,
            one => vec![one],
        }),
    )
}

/// Python's `float(value)` on a timeout written as a number or as text, so `"30"` is 30 seconds and
/// text `float` refuses is no timeout.
pub fn seconds(timeout: &Spelled<f64>) -> Option<f64> {
    match timeout {
        Spelled::Value(seconds) => Some(*seconds),
        Spelled::Text(text) => float(text),
    }
}

/// `drop_params_flag(value)`: a key that is present reads through LiteLLM's `lax_bool`, so `null` and
/// text pydantic refuses are off, and an absent key stays `None`.
pub(crate) fn lax_flag<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<bool>, D::Error> {
    Ok(Some(
        lax_bool(&PythonValue::deserialize(deserializer)?) == Some(true),
    ))
}
