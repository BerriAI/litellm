use std::{collections::BTreeMap, fmt, ops::Deref};

use serde::Deserialize;

pub type Value = serde_yaml_ng::Value;

#[derive(Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct Object(BTreeMap<String, Value>);

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

#[derive(Clone, Deserialize)]
#[serde(transparent)]
pub struct Raw(Value);

impl Raw {
    pub fn get(&self) -> &Value {
        &self.0
    }
}

impl fmt::Debug for Raw {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.0 {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Sequence(_) => "sequence",
            Value::Mapping(_) => "mapping",
            Value::Tagged(_) => "tagged",
        };
        formatter.debug_tuple("Raw").field(&kind).finish()
    }
}
