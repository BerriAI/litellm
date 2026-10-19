//! What the model cost map says about a model, for configs whose request rules depend on it
//! the way Python's `supports_*` helpers read `litellm.model_cost`.

use serde_json::{Map, Value};

pub trait ModelInfo: Send + Sync {
    /// The cost map entry `model` names, if there is one.
    fn entry(&self, model: &str) -> Option<&Map<String, Value>>;

    /// Whether any entry of the map sets `key`.
    fn declares(&self, key: &str) -> bool;
}

/// Python's `_supports_factory`: the entry sets `key` to `true`. An unknown model supports
/// nothing.
pub fn supports(info: Option<&dyn ModelInfo>, model: &str, key: &str) -> bool {
    field(info, model, key).and_then(Value::as_bool) == Some(true)
}

/// Python's `is_explicitly_disabled_factory`: the entry sets `key` to `false`. An unknown
/// model or an unset key is not disabled.
pub fn explicitly_disabled(info: Option<&dyn ModelInfo>, model: &str, key: &str) -> bool {
    field(info, model, key).and_then(Value::as_bool) == Some(false)
}

/// The value the entry for `model` sets for `key`.
pub fn field<'a>(info: Option<&'a dyn ModelInfo>, model: &str, key: &str) -> Option<&'a Value> {
    info?
        .entry(model)?
        .get(key)
        .filter(|value| !value.is_null())
}

/// Whether the map has an entry for `model`.
pub fn known(info: Option<&dyn ModelInfo>, model: &str) -> bool {
    info.is_some_and(|info| info.entry(model).is_some())
}
