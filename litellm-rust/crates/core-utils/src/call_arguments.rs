use std::ops::Deref;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallArguments(Map<String, Value>);

impl CallArguments {
    pub fn select(&self, names: &[&str]) -> Map<String, Value> {
        self.iter()
            .filter(|(name, _)| names.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid argument: {path}")]
pub struct ArgumentError {
    pub path: String,
}

pub fn parse_options<T: DeserializeOwned>(arguments: &CallArguments) -> Result<T, ArgumentError> {
    let deserializer = serde::de::value::MapDeserializer::new(
        arguments.iter().map(|(name, value)| (name.as_str(), value)),
    );
    serde_path_to_error::deserialize(deserializer).map_err(|error| ArgumentError {
        path: error.path().to_string(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgumentSpec {
    pub name: &'static str,
    pub secret: bool,
}

impl Deref for CallArguments {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Map<String, Value>> for CallArguments {
    fn from(values: Map<String, Value>) -> Self {
        Self(values)
    }
}

impl From<CallArguments> for Map<String, Value> {
    fn from(arguments: CallArguments) -> Self {
        arguments.0
    }
}

impl FromIterator<(String, Value)> for CallArguments {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for CallArguments {
    type Item = (String, Value);
    type IntoIter = serde_json::map::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn typed_views_preserve_missing_and_explicit_null_in_the_source() {
        #[derive(Deserialize)]
        struct Options {
            enabled: Option<bool>,
        }
        let arguments: CallArguments =
            serde_json::from_value(json!({"enabled":null,"future":0})).unwrap();
        assert!(
            parse_options::<Options>(&arguments)
                .unwrap()
                .enabled
                .is_none()
        );
        assert_eq!(arguments.get("enabled"), Some(&Value::Null));
        assert_eq!(arguments.get("missing"), None);
        let invalid = serde_json::from_value(json!({"enabled":0})).unwrap();
        assert_eq!(
            parse_options::<Options>(&invalid).err().unwrap().path,
            "enabled"
        );
    }
}
