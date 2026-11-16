//! `_router_weights` as `validate_router_weights` reads it (`litellm/types/router_weights.py`):
//! model group to deployment id to a finite weight of at least 0 (an int or a float, never a
//! bool or a string), with non-blank ids and at least one positive weight in every non-empty group.

use std::collections::BTreeMap;

use serde_json::Value;

/// Validated request weights.
pub type RouterWeights = BTreeMap<String, BTreeMap<String, f64>>;

/// The request's weights didn't validate; the host raises the validator's own error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidRouterWeights;

fn identifier(text: &str) -> bool {
    text.chars().any(|c| !c.is_whitespace())
}

fn weight(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|w| w.is_finite() && *w >= 0.0),
        _ => None,
    }
}

/// `validate_router_weights(raw)`: `None` when the request passed none.
pub fn validate(raw: Option<&Value>) -> Result<Option<RouterWeights>, InvalidRouterWeights> {
    let groups = match raw {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(groups)) => groups,
        Some(_) => return Err(InvalidRouterWeights),
    };
    let mut validated = RouterWeights::new();
    for (group, weights) in groups {
        let Value::Object(weights) = weights else {
            return Err(InvalidRouterWeights);
        };
        if !identifier(group) {
            return Err(InvalidRouterWeights);
        }
        let mut entries = BTreeMap::new();
        for (id, value) in weights {
            if !identifier(id) {
                return Err(InvalidRouterWeights);
            }
            entries.insert(id.clone(), weight(value).ok_or(InvalidRouterWeights)?);
        }
        if !entries.is_empty() && !entries.values().any(|w| *w > 0.0) {
            return Err(InvalidRouterWeights);
        }
        validated.insert(group.clone(), entries);
    }
    Ok(Some(validated))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::all_zero(json!({"g": {"a": 0.0, "b": 0.0}}))]
    #[case::negative(json!({"g": {"a": -1.0}}))]
    #[case::text(json!({"g": {"a": "1"}}))]
    #[case::boolean(json!({"g": {"a": true}}))]
    #[case::blank_id(json!({"g": {" ": 1.0}}))]
    #[case::not_a_mapping(json!(["g"]))]
    fn invalid_weights_are_refused(#[case] raw: Value) {
        assert_eq!(validate(Some(&raw)), Err(InvalidRouterWeights));
    }

    #[rstest]
    #[case::empty_group(json!({"g": {}}))]
    #[case::int_weight(json!({"g": {"a": 1, "b": 0}}))]
    fn valid_weights_are_kept(#[case] raw: Value) {
        assert!(validate(Some(&raw)).unwrap().is_some());
    }
}
