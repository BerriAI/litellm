pub fn parse_str_bool(value: &str) -> Option<bool> {
    let token = value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    });
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
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
}
