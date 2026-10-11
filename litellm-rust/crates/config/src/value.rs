use std::{collections::BTreeMap, fmt, ops::Deref};

use serde::{Deserialize, Deserializer};

pub type Value = serde_yaml_ng::Value;
pub type AdditionalFields = BTreeMap<String, Value>;

#[derive(Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct Object(BTreeMap<String, Value>);

impl Object {
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
