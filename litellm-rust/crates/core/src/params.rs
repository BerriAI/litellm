use std::ops::Deref;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueParams(Map<String, Value>);

impl OpaqueParams {
    pub fn into_inner(self) -> Map<String, Value> {
        self.0
    }

    pub fn retain_supported(&self, supported: &[&str]) -> Self {
        Self(
            self.iter()
                .filter(|(name, _)| supported.contains(&name.as_str()))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        )
    }
}

impl Deref for OpaqueParams {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Map<String, Value>> for OpaqueParams {
    fn from(value: Map<String, Value>) -> Self {
        Self(value)
    }
}

impl From<OpaqueParams> for Map<String, Value> {
    fn from(value: OpaqueParams) -> Self {
        value.0
    }
}

impl FromIterator<(String, Value)> for OpaqueParams {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for OpaqueParams {
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
    fn supported_keys_preserve_opaque_values() {
        let params: OpaqueParams = serde_json::from_value(json!({
            "object": {"future": [1, null]},
            "null": null,
            "unsupported": true
        }))
        .unwrap();

        let retained = params.retain_supported(&["object", "null"]);

        assert_eq!(
            serde_json::to_value(retained).unwrap(),
            json!({"object": {"future": [1, null]}, "null": null})
        );
    }

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
