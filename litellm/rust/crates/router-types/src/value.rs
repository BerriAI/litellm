use std::fmt;

use serde::Deserialize;

/// A typed value, or the text a config spells in its place: an `os.environ/NAME` reference
/// the secrets layer resolves later, or a word Python's coercion reads the value from.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Spelled<T> {
    Value(T),
    Text(String),
}

impl<T: fmt::Display> fmt::Display for Spelled<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => value.fmt(formatter),
            Self::Text(text) => formatter.write_str(text),
        }
    }
}
