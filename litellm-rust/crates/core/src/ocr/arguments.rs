use std::ops::Deref;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OcrArguments(Map<String, Value>);

impl OcrArguments {
    pub(crate) fn parse<T: DeserializeOwned>(&self) -> Result<T, super::Error> {
        super::wire::decode_request_value(Value::Object(self.0.clone()), "optional_params")
    }

    pub(crate) fn select(&self, names: &[&str]) -> Map<String, Value> {
        self.iter()
            .filter(|(name, _)| names.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    pub(crate) fn compose_body<B: Serialize>(
        &self,
        body: &B,
        consumed: &[&str],
    ) -> Result<Value, super::Error> {
        let Value::Object(fields) =
            serde_json::to_value(body).map_err(|_| crate::params::Error::Body)?
        else {
            return Err(crate::params::Error::Body.into());
        };
        let overrides = match self.get("extra_body") {
            None | Some(Value::Null) => None,
            Some(Value::Object(fields)) => Some(fields),
            Some(_) => return Err(crate::params::Error::ExtraBody.into()),
        };
        let extensions = self.iter().filter(|(name, _)| {
            !consumed.contains(&name.as_str())
                && name.as_str() != "extra_body"
                && !crate::params::is_control_param(name)
        });
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
}

impl Deref for OcrArguments {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Map<String, Value>> for OcrArguments {
    fn from(values: Map<String, Value>) -> Self {
        Self(values)
    }
}

impl From<OcrArguments> for Map<String, Value> {
    fn from(arguments: OcrArguments) -> Self {
        arguments.0
    }
}

impl FromIterator<(String, Value)> for OcrArguments {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for OcrArguments {
    type Item = (String, Value);
    type IntoIter = serde_json::map::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
