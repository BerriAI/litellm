use serde_json::Value;

/// A claim used as an id: strings as they are, integers in decimal, anything else is no id.
pub(crate) fn id_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some(number.to_string()),
        _ => None,
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

    use super::{id_text, trim};

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
    #[case::ascii(" a b \n", "a b")]
    #[case::information_separators("\u{1c}a\u{1f}", "a")]
    #[case::unicode("\u{a0}a\u{3000}", "a")]
    fn trims_separators(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(trim(text), expected);
    }
}
