//! Value helpers shared by the row builder: the Python dict, str and JSON operations from
//! `litellm-python-compat`, re-exported under the names the modules use.

use litellm_python_compat::{BigInt, Value, repr, unicode};
use num_traits::ToPrimitive;

pub(crate) use litellm_python_compat::dict::{Dict, entries, get, get_or_none, has, lookup};
pub(crate) use litellm_python_compat::pystr::{len as char_len, slice};
pub(crate) use litellm_python_compat::safe_json::{SafeJsonOptions, safe_json_structure};
pub(crate) use litellm_python_compat::truthy::{or, truthy};

use crate::RowError;

pub(crate) const NONE: Value = Value::None;

pub(crate) fn as_str(value: &Value) -> Option<&str> {
    match value {
        Value::Str(text) => Some(text),
        _ => None,
    }
}

/// `value` when it is a str with more than whitespace: `isinstance(value, str) and value.strip()`.
pub(crate) fn non_blank_str(value: &Value) -> Option<&str> {
    as_str(value).filter(|text| !text.trim_matches(unicode::isspace).is_empty())
}

/// `value is True`.
pub(crate) fn is_true(value: &Value) -> bool {
    matches!(value, Value::Bool(true))
}

pub(crate) fn text(value: &str) -> Value {
    Value::Str(value.to_owned())
}

/// `str(value)`.
pub(crate) fn to_str(value: &Value) -> String {
    repr::to_str(value)
}

pub(crate) fn big_to_f64(value: &BigInt) -> Option<f64> {
    value.to_f64().filter(|number| number.is_finite())
}

/// LiteLLM's `safe_dumps(value, non_finite_as_null=...)`, which decides what every JSON column of
/// the row holds.
pub(crate) fn safe_dumps(value: &Value, non_finite_as_null: bool) -> Result<String, RowError> {
    Ok(litellm_python_compat::safe_json::safe_dumps(
        value,
        non_finite_as_null,
    )?)
}
