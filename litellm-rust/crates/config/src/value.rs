use std::{collections::BTreeMap, fmt, ops::Deref};

use serde::Deserialize;

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

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum NumberOrString {
    Number(f64),
    String(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Flag {
    Boolean(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    Many(Box<[T]>),
    One(T),
}

impl<T> OneOrMany<T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Many(values) => values.len(),
            Self::One(_) => 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Many(values) => values.is_empty(),
            Self::One(_) => false,
        }
    }
}
