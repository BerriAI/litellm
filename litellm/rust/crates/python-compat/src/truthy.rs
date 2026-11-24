//! Python truthiness: `bool(value)` and the `a or b` operator over [`Value`].

use num_bigint::Sign;

use crate::Value;

/// `bool(value)` for plain data: `None`, `False`, zero, and empty containers are false.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::None => false,
        Value::Bool(value) => *value,
        Value::Int(value) => value.sign() != Sign::NoSign,
        Value::Float(value) => *value != 0.0,
        Value::Complex { re, im } => *re != 0.0 || *im != 0.0,
        Value::Str(value) => !value.is_empty(),
        Value::Bytes(value) => !value.is_empty(),
        Value::Tuple(values) | Value::List(values) | Value::Set(values) => !values.is_empty(),
        Value::Dict(entries) => !entries.is_empty(),
    }
}

/// `bool(json.loads(text))`: [`truthy`] of what [`crate::json::from_json`] returns, read
/// straight off the `serde_json` value.
pub fn json_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(number) => number.as_f64().is_none_or(|value| value != 0.0),
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(entries) => !entries.is_empty(),
    }
}

/// Python's `a or b or ...` over optional text: the first non-empty value, else the last value,
/// which may be empty or `None`.
pub fn first_truthy(values: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    values.into_iter().fold(None, |chosen, value| match chosen {
        Some(chosen) if !chosen.is_empty() => Some(chosen),
        _ => value,
    })
}

/// `first or second`: the first value when it is truthy, else the second.
pub fn or<'a>(first: &'a Value, second: &'a Value) -> &'a Value {
    if truthy(first) { first } else { second }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::first_truthy;

    #[rstest]
    #[case::first_non_empty(&[Some("a"), Some("b")], Some("a"))]
    #[case::skips_empty_and_missing(&[Some(""), None, Some("c")], Some("c"))]
    #[case::all_falsy_keeps_the_last_empty(&[None, Some("")], Some(""))]
    #[case::all_falsy_keeps_the_last_missing(&[Some(""), None], None)]
    #[case::nothing(&[], None)]
    fn first_truthy_is_python_s_or_chain(
        #[case] values: &[Option<&str>],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            first_truthy(values.iter().map(|value| value.map(str::to_owned))).as_deref(),
            expected
        );
    }
}
