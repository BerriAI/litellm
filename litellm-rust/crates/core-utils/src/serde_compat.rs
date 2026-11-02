fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}

pub fn parse_str_bool(value: &str) -> Option<bool> {
    let token = python_strip(value);
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
}

pub fn parse_pydantic_bool(value: &str) -> Option<bool> {
    match python_strip(value).to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
        _ => None,
    }
}

/// `redis-py` string Booleans: only `1`, `true`, and `yes` (case-insensitive) are true.
pub fn parse_redis_bool(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::trimmed_true(" True ", Some(true))]
    #[case::control_whitespace_true("\u{1c}TRUE\u{1f}", Some(true))]
    #[case::unicode_whitespace_false("\u{a0}False\u{2003}", Some(false))]
    #[case::zero_width_space("true\u{200b}", None)]
    #[case::yes("yes", None)]
    #[case::one("1", None)]
    #[case::empty("", None)]
    #[case::unknown("unknown", None)]
    fn boolean_tokens_follow_python_string_trimming_without_redis_tokens(
        #[case] input: &str,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(parse_str_bool(input), expected, "{input:?}");
    }

    #[rstest]
    #[case::yes(" YES ", Some(true))]
    #[case::on("On", Some(true))]
    #[case::one("1", Some(true))]
    #[case::t("t", Some(true))]
    #[case::y("y", Some(true))]
    #[case::control_whitespace_true("\u{1c}TRUE\u{1f}", Some(true))]
    #[case::no("No", Some(false))]
    #[case::off("OFF", Some(false))]
    #[case::zero("0", Some(false))]
    #[case::f("F", Some(false))]
    #[case::n("n", Some(false))]
    #[case::false_word("false", Some(false))]
    #[case::two("2", None)]
    #[case::empty("", None)]
    #[case::unknown("maybe", None)]
    fn boolean_words_follow_pydantic_lax_bool_parsing(
        #[case] input: &str,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(parse_pydantic_bool(input), expected, "{input:?}");
    }
}
