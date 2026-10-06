use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{Map, Value};

use super::CacheTarget;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKeyInput {
    Preset(String),
    Request {
        target: CacheTarget,
        parameters: Value,
    },
}

impl CacheKeyInput {
    pub fn request(target: CacheTarget, parameters: Value) -> Self {
        let parameters = match canonical(parameters) {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .filter(|(_, value)| !value.is_null())
                    .collect(),
            ),
            value => value,
        };
        Self::Request { target, parameters }
    }

    pub fn forwarded(
        target: CacheTarget,
        parameters: impl IntoIterator<Item = (String, Value)>,
        headers: impl IntoIterator<Item = (&'static str, Value)>,
    ) -> Self {
        Self::request(
            target,
            Value::Object(
                parameters
                    .into_iter()
                    .filter(|(name, _)| name != "model")
                    .chain(
                        headers
                            .into_iter()
                            .map(|(name, value)| (name.to_owned(), value)),
                    )
                    .collect(),
            ),
        )
    }
}

pub fn extra_headers(headers: Option<&Map<String, Value>>) -> Option<(&'static str, Value)> {
    headers.map(|headers| {
        (
            "extra_headers",
            Value::Object(
                headers
                    .iter()
                    .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
                    .collect(),
            ),
        )
    })
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(name, value)| (name, canonical(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        value => value,
    }
}
