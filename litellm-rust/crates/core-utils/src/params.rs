use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueParams(Map<String, Value>);

impl Deref for OpaqueParams {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OpaqueParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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

    use super::OpaqueParams;

    #[test]
    fn outer_value_must_be_an_object() {
        assert!(serde_json::from_value::<OpaqueParams>(json!(["value"])).is_err());
    }
}
