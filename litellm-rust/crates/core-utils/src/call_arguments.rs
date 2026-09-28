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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderParameters {
    fields: Map<String, Value>,
    overrides: Map<String, Value>,
}

impl ProviderParameters {
    pub fn from_arguments(arguments: &CallArguments) -> Result<Self, crate::params::Error> {
        let overrides = match arguments.get("extra_body") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(fields)) => fields
                .iter()
                .filter(|(name, _)| is_provider_field(name))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            Some(_) => return Err(crate::params::Error::ExtraBody),
        };
        let fields = arguments
            .iter()
            .filter(|(name, _)| is_provider_field(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        Ok(Self { fields, overrides })
    }

    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }

    pub fn overrides(&self) -> &Map<String, Value> {
        &self.overrides
    }

    pub fn validate_stream(&self, stream: Option<bool>) -> Result<(), crate::params::Error> {
        if self
            .overrides
            .get("stream")
            .is_some_and(|value| *value != Value::Bool(stream.unwrap_or(false)))
        {
            return Err(crate::params::Error::ProtectedField { field: "stream" });
        }
        Ok(())
    }

    pub fn compose<B: Serialize>(
        &self,
        body: &B,
        consumed: &[&str],
    ) -> Result<Value, crate::params::Error> {
        let Value::Object(fields) =
            serde_json::to_value(body).map_err(|_| crate::params::Error::Body)?
        else {
            return Err(crate::params::Error::Body);
        };
        self.validate_stream(fields.get("stream").and_then(Value::as_bool))?;
        Ok(Value::Object(
            fields
                .into_iter()
                .chain(
                    self.fields
                        .iter()
                        .filter(|(name, _)| !consumed.contains(&name.as_str()))
                        .chain(self.overrides.iter())
                        .filter(|(name, _)| name.as_str() != "model" && is_provider_field(name))
                        .map(|(name, value)| (name.clone(), value.clone())),
                )
                .collect(),
        ))
    }
}

fn is_provider_field(name: &str) -> bool {
    !matches!(name, "model" | "extra_body") && !crate::params::is_control_param(name)
}

pub fn compose_body<B: Serialize>(
    arguments: &CallArguments,
    body: &B,
    consumed: &[&str],
) -> Result<Value, crate::params::Error> {
    ProviderParameters::from_arguments(arguments)?.compose(body, consumed)
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::credential("api_key")]
    #[case::callbacks("callbacks")]
    #[case::context("proxy_server_request")]
    #[case::internal("litellm_future_control")]
    #[case::private("_litellm_future_control")]
    #[case::endpoint("custom_endpoint")]
    fn provider_parameters_separate_controls_without_changing_nested_data(#[case] name: &str) {
        let future = json!({name: "provider-owned", "values": [false, 0, null, "", []]});
        let arguments: CallArguments = serde_json::from_value(json!({
            name: "internal", "future": future, "explicit_null": null,
            "extra_body": {name: "internal override", "future": {"replacement": false}}
        }))
        .unwrap();
        let params = ProviderParameters::from_arguments(&arguments).unwrap();
        assert_eq!(
            params.fields(),
            &json!({"future": future, "explicit_null": null})
                .as_object()
                .unwrap()
                .clone()
        );
        assert_eq!(
            params.compose(&json!({"model": "resolved"}), &[]).unwrap(),
            json!({
                "model": "resolved", "future": {"replacement": false}, "explicit_null": null
            })
        );
        assert_eq!(arguments[name], "internal");
    }

    #[rstest]
    #[case::enable(json!({}), json!(true))]
    #[case::disable(json!({"stream": true}), json!(false))]
    #[case::invalid(json!({"stream": false}), json!("yes"))]
    fn overrides_cannot_change_the_selected_delivery(#[case] body: Value, #[case] stream: Value) {
        let arguments = serde_json::from_value(json!({"extra_body": {"stream": stream}})).unwrap();
        assert_eq!(
            compose_body(&arguments, &body, &[]),
            Err(crate::params::Error::ProtectedField { field: "stream" })
        );
    }

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

    #[rstest]
    #[case::boolean(json!(false))]
    #[case::number(json!(0))]
    #[case::array(json!([]))]
    #[case::string(json!(""))]
    fn invalid_extra_body_is_rejected_without_coercing_it_to_empty(
        #[case] value: serde_json::Value,
    ) {
        let arguments = serde_json::from_value(json!({"extra_body":value})).unwrap();
        assert_eq!(
            compose_body(&arguments, &json!({}), &[]),
            Err(crate::params::Error::ExtraBody)
        );
    }

    #[rstest]
    #[case::null(json!(null), json!({}))]
    fn null_extra_body_is_coerced_to_empty_object(
        #[case] value: serde_json::Value,
        #[case] expected: serde_json::Value,
    ) {
        let arguments = serde_json::from_value(json!({"extra_body":value})).unwrap();
        assert_eq!(compose_body(&arguments, &json!({}), &[]).unwrap(), expected);
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
