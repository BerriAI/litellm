use litellm_host::raw_json;
use serde_json::Value;

use crate::{
    Value as PythonValue,
    json::{from_json, loads},
    repr::to_str,
};

pub fn python_value(value: &Value) -> PythonValue {
    match value {
        Value::String(text) => raw_json::decode(text)
            .and_then(|raw| loads(raw).ok())
            .unwrap_or_else(|| PythonValue::Str(text.clone())),
        Value::Array(items) => PythonValue::List(items.iter().map(python_value).collect()),
        Value::Object(entries) => PythonValue::Dict(
            entries
                .iter()
                .map(|(key, item)| (PythonValue::Str(key.clone()), python_value(item)))
                .collect(),
        ),
        Value::Null | Value::Bool(_) | Value::Number(_) => from_json(value.clone()),
    }
}

pub fn python_number(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        Value::Number(number) => number.as_f64(),
        Value::String(text) => match raw_json::decode(text)? {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            raw => raw.parse().ok(),
        },
        _ => None,
    }
}

pub fn known_str<T: serde::Serialize>(known: &T) -> String {
    serde_json::to_value(known)
        .map(|known| to_str(&from_json(known)))
        .unwrap_or_default()
}

pub fn value_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(text) => match raw_json::decode(text) {
            Some("NaN") => "nan".into(),
            Some("Infinity") => "inf".into(),
            Some("-Infinity") => "-inf".into(),
            Some(raw) => raw.trim_matches('"').into(),
            None => text.clone(),
        },
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::int(json!(3), Some(3.0))]
    #[case::float(json!(1.5), Some(1.5))]
    #[case::bool(json!(true), Some(1.0))]
    #[case::wide(json!(raw_json::encode("18446744073709551616")), Some(18_446_744_073_709_551_616.0))]
    #[case::negative_infinity(json!(raw_json::encode("-Infinity")), Some(f64::NEG_INFINITY))]
    #[case::string(json!("1"), None)]
    #[case::marked_string(json!(raw_json::encode("\"1\"")), None)]
    #[case::null(Value::Null, None)]
    fn numbers_compare_as_python_compares_them(
        #[case] value: Value,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(python_number(&value), expected);
    }

    #[test]
    fn nan_is_a_number_unequal_to_everything() {
        let nan = json!(raw_json::encode("NaN"));
        assert!(python_number(&nan).is_some_and(f64::is_nan));
        assert_ne!(python_number(&nan), Some(1.0));
    }

    #[rstest]
    #[case::half(0.5, "0.5")]
    #[case::wide(1e16, "1e+16")]
    fn known_values_print_as_python_prints_them(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(known_str(&value), expected);
    }

    #[rstest]
    #[case::int(json!(0), "0")]
    #[case::bool(json!(false), "False")]
    #[case::string(json!("0.5"), "0.5")]
    #[case::nan(json!(raw_json::encode("NaN")), "nan")]
    #[case::wide(json!(raw_json::encode("-1180591620717411303424")), "-1180591620717411303424")]
    fn values_print_as_python_prints_them(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(value_str(&value), expected);
    }
}
