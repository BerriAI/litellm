/// LiteLLM's `str_to_bool(text)`: only `true` and `false` after `str.strip()`, ignoring case.
pub fn str_to_bool(text: &str) -> Option<bool> {
    let token = strip(text);
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
}

/// LiteLLM's `lax_bool(text)`: pydantic's lax `bool` validation of `text.strip()`.
pub fn lax_bool(text: &str) -> Option<bool> {
    match strip(text).to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
        _ => None,
    }
}

/// `str.strip()`. Python's `str.isspace` is Unicode `White_Space` plus the ASCII information separators.
fn strip(text: &str) -> &str {
    text.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{lax_bool, str_to_bool};

    // Expected values are CPython 3.14.7 `text.strip().lower()` against {"true"} and {"false"}
    #[rstest]
    #[case::trimmed_true(" True ", Some(true))]
    #[case::tab_and_newline("\ttrue\n", Some(true))]
    #[case::information_separators("\u{1c}TRUE\u{1f}", Some(true))]
    #[case::next_line("\u{85}true", Some(true))]
    #[case::unicode_spaces_false("\u{a0}False\u{2003}", Some(false))]
    #[case::zero_width_space("true\u{200b}", None)]
    #[case::trailing_nul("true\0", None)]
    #[case::yes("yes", None)]
    #[case::one("1", None)]
    #[case::empty("", None)]
    #[case::unknown("unknown", None)]
    fn str_to_bool_reads_like_litellm(#[case] text: &str, #[case] expected: Option<bool>) {
        assert_eq!(str_to_bool(text), expected);
    }

    // Expected values are pydantic 2.12.5 `TypeAdapter(bool).validate_python(text.strip())` on CPython 3.14.7
    #[rstest]
    #[case::trimmed_yes(" YES ", Some(true))]
    #[case::on("On", Some(true))]
    #[case::one("1", Some(true))]
    #[case::t("t", Some(true))]
    #[case::y("y", Some(true))]
    #[case::information_separators("\u{1c}TRUE\u{1f}", Some(true))]
    #[case::no("No", Some(false))]
    #[case::off("OFF", Some(false))]
    #[case::zero("0", Some(false))]
    #[case::f("F", Some(false))]
    #[case::n("n", Some(false))]
    #[case::false_word("false", Some(false))]
    #[case::two("2", None)]
    #[case::leading_zero("01", None)]
    #[case::float_text("1.0", None)]
    #[case::zero_width_space("true\u{200b}", None)]
    #[case::empty("", None)]
    #[case::unknown("maybe", None)]
    fn lax_bool_reads_like_litellm(#[case] text: &str, #[case] expected: Option<bool>) {
        assert_eq!(lax_bool(text), expected);
    }
}
