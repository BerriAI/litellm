/// `repr()` of a Python `str`: single quotes unless the text holds a single quote and no
/// double quote, with backslashes, the chosen quote and control characters escaped.
pub fn python_str_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let escaped: String = value
        .chars()
        .map(|character| match character {
            '\\' => "\\\\".to_string(),
            '\t' => "\\t".to_string(),
            '\n' => "\\n".to_string(),
            '\r' => "\\r".to_string(),
            character if character == quote => format!("\\{character}"),
            character
                if (character as u32) < 0x20 || (0x7f..0xa0).contains(&(character as u32)) =>
            {
                format!("\\x{:02x}", character as u32)
            }
            character => character.to_string(),
        })
        .collect();
    format!("{quote}{escaped}{quote}")
}

/// `repr()` of the Python value a JSON value decodes to.
pub fn python_value_repr(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => python_str_repr(text),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(python_value_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_str_repr(key),
                    python_value_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{python_str_repr, python_value_repr};

    #[rstest::rstest]
    #[case::null(json!(null), "None")]
    #[case::true_(json!(true), "True")]
    #[case::false_(json!(false), "False")]
    #[case::integer(json!(5), "5")]
    #[case::float(json!(1.5), "1.5")]
    #[case::string(json!("it's"), "\"it's\"")]
    #[case::list(json!(["a", 1]), "['a', 1]")]
    #[case::dict(json!({"format": "native"}), "{'format': 'native'}")]
    #[case::empty_list(json!([]), "[]")]
    fn value_repr_matches_python(#[case] value: serde_json::Value, #[case] expected: &str) {
        assert_eq!(python_value_repr(&value), expected);
    }

    #[rstest::rstest]
    #[case::plain("native", "'native'")]
    #[case::single_quote("it's", "\"it's\"")]
    #[case::both_quotes("it's \"x\"", "'it\\'s \"x\"'")]
    #[case::double_quote("say \"x\"", "'say \"x\"'")]
    #[case::backslash("a\\b", "'a\\\\b'")]
    #[case::whitespace("a\tb\nc\rd", "'a\\tb\\nc\\rd'")]
    #[case::control("a\u{1}b\u{7f}c\u{85}", "'a\\x01b\\x7fc\\x85'")]
    #[case::unicode("café", "'café'")]
    #[case::empty("", "''")]
    fn matches_python_repr(#[case] value: &str, #[case] expected: &str) {
        assert_eq!(python_str_repr(value), expected);
    }
}
