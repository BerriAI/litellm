use std::ops::Deref;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallArguments(Map<String, Value>);

impl CallArguments {
    pub(crate) fn select(&self, names: &[&str]) -> Map<String, Value> {
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

pub fn compose_body<B: Serialize>(
    arguments: &CallArguments,
    body: &B,
    consumed: &[&str],
) -> Result<Value, crate::params::Error> {
    let Value::Object(fields) =
        serde_json::to_value(body).map_err(|_| crate::params::Error::Body)?
    else {
        return Err(crate::params::Error::Body);
    };
    let overrides = match arguments.get("extra_body") {
        None | Some(Value::Null) => None,
        Some(Value::Object(fields)) => Some(fields),
        Some(_) => return Err(crate::params::Error::ExtraBody),
    };
    let extensions = arguments
        .iter()
        .filter(|(name, _)| !consumed.contains(&name.as_str()));
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                extensions
                    .chain(overrides.into_iter().flatten())
                    .filter(|(name, _)| {
                        name.as_str() != "model"
                            && name.as_str() != "extra_body"
                            && !crate::params::is_control_param(name)
                    })
                    .map(|(name, value)| (name.clone(), value.clone())),
            )
            .collect(),
    ))
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
    fn composition_preserves_extensions_and_applies_shallow_explicit_overrides() {
        let original = json!({
            "known": false, "future": {"old": 1}, "null": null, "zero": 0,
            "metadata": {"host": true}, "timeout": 30, "api_key": "secret",
            "extra_body": {
                "known": null, "future": {"new": [false, 0, null]},
                "metadata": {"provider": true}, "model": "ignored", "api_key": "ignored"
            }
        });
        let arguments = serde_json::from_value(original.clone()).unwrap();
        let body = compose_body(
            &arguments,
            &json!({"model":"resolved", "known":false}),
            &["known"],
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model":"resolved", "known":null, "future":{"new":[false,0,null]},
                "null":null, "zero":0, "metadata":{"provider":true}
            })
        );
        assert_eq!(serde_json::to_value(arguments).unwrap(), original);
    }

    #[test]
    fn invalid_extra_body_is_rejected_without_coercing_it_to_empty() {
        for value in [json!(false), json!(0), json!([]), json!("")] {
            let arguments = serde_json::from_value(json!({"extra_body":value})).unwrap();
            assert_eq!(
                compose_body(&arguments, &json!({}), &[]),
                Err(crate::params::Error::ExtraBody)
            );
        }
        let arguments = serde_json::from_value(json!({"extra_body":null})).unwrap();
        assert_eq!(
            compose_body(&arguments, &json!({}), &[]).unwrap(),
            json!({})
        );
    }

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
