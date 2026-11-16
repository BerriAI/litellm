use num_bigint::Sign;

use crate::{Value, json::from_json};

/// `bool(value)` for a field read from JSON, where a missing field is `None`.
pub fn json_truthy(value: Option<&serde_json::Value>) -> bool {
    value.is_some_and(|value| truthy(&from_json(value.clone())))
}

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

#[cfg(test)]
mod tests {
    #[rstest::rstest]
    #[case::missing(None, false)]
    #[case::null(Some(serde_json::json!(null)), false)]
    #[case::empty_text(Some(serde_json::json!("")), false)]
    #[case::zero(Some(serde_json::json!(0)), false)]
    #[case::zero_float(Some(serde_json::json!(0.0)), false)]
    #[case::empty_dict(Some(serde_json::json!({})), false)]
    #[case::text(Some(serde_json::json!("x")), true)]
    #[case::list(Some(serde_json::json!([0])), true)]
    fn a_json_field_is_truthy_as_python_reads_it(
        #[case] value: Option<serde_json::Value>,
        #[case] expected: bool,
    ) {
        assert_eq!(super::json_truthy(value.as_ref()), expected);
    }
}
