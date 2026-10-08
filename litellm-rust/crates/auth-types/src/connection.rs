use std::{collections::BTreeMap, fmt};

use serde_json::{Map, Value};

use crate::{
    InputSource,
    fields::{ALL, ConnectionField},
};

fn connection_field(name: &str) -> Option<&'static ConnectionField> {
    ALL.iter()
        .flat_map(|fields| fields.iter().copied())
        .find(|field| field.kwargs.contains(&name))
}

pub fn is_connection_name(name: &str) -> bool {
    connection_field(name).is_some()
}

pub fn is_secret_connection_name(name: &str) -> bool {
    connection_field(name).is_some_and(|field| field.secret)
}

/// What a provider needs to connect for one call (credentials and endpoint settings), keyed
/// by their LiteLLM argument names. Built only by keeping the connection names out of a
/// call's arguments, so a provider parameter can never be read as a credential and a
/// credential never reaches a request body.
#[derive(Clone, Default, PartialEq)]
pub struct ConnectionArguments {
    values: Map<String, Value>,
    sources: BTreeMap<String, InputSource>,
}

impl ConnectionArguments {
    pub fn from_arguments<'a>(
        arguments: impl IntoIterator<Item = (&'a String, &'a Value)>,
        source: impl Fn(&str) -> InputSource,
    ) -> Self {
        let values: Map<String, Value> = arguments
            .into_iter()
            .filter(|(name, _)| is_connection_name(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let sources = values
            .keys()
            .map(|name| (name.clone(), source(name)))
            .collect();
        Self { values, sources }
    }

    /// Splits one call's arguments into the provider parameters and the connection
    /// arguments, so neither side can see the other's names.
    pub fn split(
        arguments: Map<String, Value>,
        source: impl Fn(&str) -> InputSource,
    ) -> (Map<String, Value>, Self) {
        let connection = Self::from_arguments(&arguments, source);
        let params = arguments
            .into_iter()
            .filter(|(name, _)| !is_connection_name(name))
            .collect();
        (params, connection)
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn source(&self, name: &str) -> InputSource {
        self.sources.get(name).copied().unwrap_or_default()
    }

    pub fn secret_names(&self) -> impl Iterator<Item = &str> {
        self.values
            .keys()
            .map(String::as_str)
            .filter(|name| is_secret_connection_name(name))
    }
}

impl fmt::Debug for ConnectionArguments {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_set().entries(self.values.keys()).finish()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn arguments(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn only_credential_names_are_kept_with_their_sources() {
        let call = arguments(json!({
            "aws_region_name": "us-east-1",
            "vertex_credentials": "{}",
            "temperature": 0.2,
            "api_key": "sk",
        }));
        let credentials = ConnectionArguments::from_arguments(&call, |name| {
            if name == "vertex_credentials" {
                InputSource::Request
            } else {
                InputSource::Deployment
            }
        });
        assert_eq!(
            credentials.get("aws_region_name"),
            Some(&json!("us-east-1"))
        );
        assert_eq!(credentials.get("temperature"), None);
        assert_eq!(credentials.get("api_key"), None);
        assert_eq!(
            credentials.source("vertex_credentials"),
            InputSource::Request
        );
        assert_eq!(
            credentials.source("aws_region_name"),
            InputSource::Deployment
        );
    }

    #[test]
    fn split_leaves_no_connection_name_in_the_params() {
        let call = arguments(json!({
            "aws_secret_access_key": "secret",
            "custom_endpoint": true,
            "temperature": 0.2,
        }));
        let (params, connection) = ConnectionArguments::split(call, |_| InputSource::Request);
        assert_eq!(Value::Object(params), json!({"temperature": 0.2}));
        assert_eq!(
            connection.get("aws_secret_access_key"),
            Some(&json!("secret"))
        );
        assert_eq!(connection.get("custom_endpoint"), Some(&json!(true)));
        assert_eq!(connection.source("custom_endpoint"), InputSource::Request);
    }

    #[test]
    fn secret_names_are_the_secret_credentials_present() {
        let call = arguments(json!({
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "client_secret": "secret",
        }));
        let credentials = ConnectionArguments::from_arguments(&call, |_| InputSource::Request);
        let mut names = credentials.secret_names().collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(names, ["aws_secret_access_key", "client_secret"]);
    }

    #[test]
    fn debug_lists_names_without_values() {
        let call = arguments(json!({"aws_secret_access_key": "do-not-print"}));
        let credentials = ConnectionArguments::from_arguments(&call, |_| InputSource::Request);
        let rendered = format!("{credentials:?}");
        assert!(rendered.contains("aws_secret_access_key"));
        assert!(!rendered.contains("do-not-print"));
    }
}
