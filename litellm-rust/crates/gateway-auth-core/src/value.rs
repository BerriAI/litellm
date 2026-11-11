use serde_json::Value;

/// A claim counts as set unless it is null, false, zero, or an empty string, array or object.
pub(crate) fn is_set(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// A claim used as an id: strings as they are, integers in decimal, anything else is no id.
pub(crate) fn id_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some(number.to_string()),
        _ => None,
    }
}

/// A value quoted in a message: strings as they are, anything else as JSON.
pub(crate) fn display(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A NumericDate claim (RFC 7519) in whole seconds. Only JSON numbers qualify.
pub(crate) fn seconds(value: &Value) -> Option<f64> {
    let Value::Number(number) = value else {
        return None;
    };
    number
        .as_i64()
        .map(|int| int as f64)
        .or_else(|| number.as_u64().map(|int| int as f64))
        .or_else(|| {
            number
                .as_f64()
                .filter(|float| float.is_finite())
                .map(f64::trunc)
        })
}

/// The JSON type name a message reports for a value.
pub(crate) fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Unicode whitespace plus the ASCII information separators U+001C to U+001F.
pub(crate) fn is_separator(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub(crate) fn trim(text: &str) -> &str {
    text.trim_matches(is_separator)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{id_text, is_set, seconds, trim};

    #[rstest]
    #[case::null(json!(null), false)]
    #[case::falsy_bool(json!(false), false)]
    #[case::zero(json!(0), false)]
    #[case::zero_float(json!(0.0), false)]
    #[case::empty_string(json!(""), false)]
    #[case::empty_array(json!([]), false)]
    #[case::empty_object(json!({}), false)]
    #[case::truthy_bool(json!(true), true)]
    #[case::number(json!(-1), true)]
    #[case::string(json!("a"), true)]
    #[case::array(json!([null]), true)]
    #[case::object(json!({"a": null}), true)]
    fn set_claims(#[case] value: Value, #[case] expected: bool) {
        assert_eq!(is_set(&value), expected);
    }

    #[rstest]
    #[case::string(json!("u-1"), Some("u-1"))]
    #[case::int(json!(12345), Some("12345"))]
    #[case::negative(json!(-7), Some("-7"))]
    #[case::large(json!(u64::MAX), Some("18446744073709551615"))]
    #[case::float(json!(1.5), None)]
    #[case::bool(json!(true), None)]
    #[case::object(json!({"a": 1}), None)]
    #[case::array(json!(["a"]), None)]
    #[case::null(json!(null), None)]
    fn ids(#[case] value: Value, #[case] expected: Option<&str>) {
        assert_eq!(id_text(&value).as_deref(), expected);
    }

    #[rstest]
    #[case::int(json!(1_700_000_000), Some(1_700_000_000.0))]
    #[case::float_truncates(json!(1_700_000_000.9), Some(1_700_000_000.0))]
    #[case::negative_float_truncates(json!(-1.5), Some(-1.0))]
    #[case::digit_string(json!("1700000000"), None)]
    #[case::bool(json!(true), None)]
    #[case::null(json!(null), None)]
    fn numeric_dates(#[case] value: Value, #[case] expected: Option<f64>) {
        assert_eq!(seconds(&value), expected);
    }

    #[rstest]
    #[case::ascii(" a b \n", "a b")]
    #[case::information_separators("\u{1c}a\u{1f}", "a")]
    #[case::unicode("\u{a0}a\u{3000}", "a")]
    fn trims_separators(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(trim(text), expected);
    }
}
