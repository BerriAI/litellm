use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Index, Range};

use base64::Engine;
use bytes::Bytes;
use bytestring::ByteString;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::Error;

#[derive(Clone, PartialEq, Eq)]
pub struct SharedText(ByteString);

impl SharedText {
    pub fn new(bytes: Bytes) -> Result<Self, std::str::Utf8Error> {
        ByteString::try_from(bytes).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn bytes(&self) -> &Bytes {
        self.0.as_bytes()
    }

    pub fn slice(&self, range: Range<usize>) -> Result<Self, Error> {
        if self.as_str().get(range.clone()).is_none() {
            return Err(Error::InvalidRequest("invalid shared text range".into()));
        }
        Ok(Self(self.0.slice_ref(&self.0[range])))
    }
}

impl From<String> for SharedText {
    fn from(value: String) -> Self {
        Self(ByteString::from(value))
    }
}

impl fmt::Debug for SharedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedText")
            .field("len", &self.0.len())
            .finish()
    }
}

impl Serialize for SharedText {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SharedText {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

#[derive(Clone, PartialEq)]
pub enum JsonPayload {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(SharedText),
    Base64(Bytes),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

impl JsonPayload {
    pub fn object<const N: usize>(fields: [(&str, Self); N]) -> Self {
        Self::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Base64(_) => "bytes",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        self.as_text().map(SharedText::as_str)
    }

    pub fn as_text(&self) -> Option<&SharedText> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Self>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn is_object(&self) -> bool {
        self.as_object().is_some()
    }

    pub fn get(&self, key: &str) -> Option<&Self> {
        self.as_object()?.get(key)
    }

    pub fn into_object(self) -> Result<BTreeMap<String, Self>, Error> {
        match self {
            Self::Object(value) => Ok(value),
            other => Err(Error::InvalidType {
                expected: "object",
                actual: other.type_name(),
            }),
        }
    }

    pub fn contains_media(&self) -> bool {
        match self {
            Self::Base64(_) => true,
            Self::String(value) => value.as_str().starts_with("data:"),
            Self::Array(items) => items.iter().any(Self::contains_media),
            Self::Object(fields) => {
                fields.get("type").and_then(Self::as_str) == Some("base64")
                    || fields.contains_key("audio")
                    || fields.contains_key("base64Source")
                    || fields.values().any(Self::contains_media)
            }
            _ => false,
        }
    }

    pub fn materialize(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.as_str().to_owned()),
            Self::Base64(value) => {
                Value::String(base64::engine::general_purpose::STANDARD.encode(value))
            }
            Self::Array(items) => Value::Array(items.iter().map(Self::materialize).collect()),
            Self::Object(fields) => Value::Object(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.materialize()))
                    .collect(),
            ),
        }
    }
}

impl fmt::Debug for JsonPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonPayload")
            .field("kind", &self.type_name())
            .finish()
    }
}

impl From<Value> for JsonPayload {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value.into()),
            Value::Array(items) => Self::Array(items.into_iter().map(Self::from).collect()),
            Value::Object(fields) => Self::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, value.into()))
                    .collect(),
            ),
        }
    }
}

impl From<String> for JsonPayload {
    fn from(value: String) -> Self {
        Self::String(value.into())
    }
}

impl From<&str> for JsonPayload {
    fn from(value: &str) -> Self {
        value.to_owned().into()
    }
}

impl Serialize for JsonPayload {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => value.serialize(serializer),
            Self::Base64(value) => serializer.collect_str(&base64::display::Base64Display::new(
                value,
                &base64::engine::general_purpose::STANDARD,
            )),
            Self::Array(items) => items.serialize(serializer),
            Self::Object(fields) => fields.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for JsonPayload {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Value::deserialize(deserializer).map(Self::from)
    }
}

impl Index<&str> for JsonPayload {
    type Output = Self;
    fn index(&self, key: &str) -> &Self {
        self.get(key).unwrap_or(&Self::Null)
    }
}

impl Index<usize> for JsonPayload {
    type Output = Self;
    fn index(&self, index: usize) -> &Self {
        self.as_array()
            .and_then(|items| items.get(index))
            .unwrap_or(&Self::Null)
    }
}

impl PartialEq<Value> for JsonPayload {
    fn eq(&self, other: &Value) -> bool {
        self.materialize() == *other
    }
}

impl PartialEq<&str> for JsonPayload {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}
