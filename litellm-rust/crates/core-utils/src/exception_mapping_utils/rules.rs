use std::sync::LazyLock;

use fancy_regex::Regex;
use serde_json::Value;

use super::Mapping;
use super::public::PublicError;

/// One text branch of a Python `_map_*_exception` function: when it applies, the class it
/// raises, and any help text appended to the message.
pub(super) struct Rule {
    pub(super) when: fn(&Mapping) -> bool,
    pub(super) error: PublicError,
    pub(super) hint: &'static str,
}

impl Rule {
    pub(super) const fn new(when: fn(&Mapping) -> bool, error: PublicError) -> Self {
        Self {
            when,
            error,
            hint: "",
        }
    }
}

/// The first rule that applies decides the class, as the `if`/`elif` chain does in Python.
pub(super) fn first_match<'r>(rules: &'r [Rule], mapping: &Mapping) -> Option<&'r Rule> {
    rules.iter().find(|rule| (rule.when)(mapping))
}

pub(super) fn contains_any(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

static STANDALONE_429: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b429\b").expect("valid regex"));
static RATE_LIMIT_PHRASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"rate[\s_\-]*limit").expect("valid regex"));

/// `ExceptionCheckers.is_error_str_rate_limit`.
pub(super) fn is_rate_limit(error_str: &str, status: Option<u16>) -> bool {
    if STANDALONE_429.is_match(error_str).unwrap_or(false) && matches!(status, None | Some(429)) {
        return true;
    }
    let lower = error_str.to_lowercase();
    RATE_LIMIT_PHRASE.is_match(&lower).unwrap_or(false)
        || lower.contains("service tier capacity exceeded")
}

/// `ExceptionCheckers.is_error_str_context_window_exceeded`.
pub(super) fn is_context_window_exceeded(error_str: &str) -> bool {
    let lower = error_str.to_lowercase();
    if lower.contains("string_above_max_length") {
        return false;
    }
    if lower.contains("invalid 'user'") && lower.contains("string too long") {
        return false;
    }
    contains_any(
        &lower,
        &[
            "exceed context limit",
            "this model's maximum context length is",
            "string too long. expected a string with maximum length",
            "model's maximum context limit",
            "is longer than the model's context length",
            "input tokens exceed the configured limit",
            "`inputs` tokens + `max_new_tokens` must be",
            "exceeds the available context size",
            "exceeds the maximum number of tokens allowed",
        ],
    ) || (lower.contains("current length is") && lower.contains("while limit is"))
        || (lower.contains("maximum input length is") && lower.contains("tokens"))
}

/// The integer `error.code` of a JSON error body, read the way Python's `int()` would.
pub(super) fn body_error_code(error_str: &str) -> Option<i64> {
    let body: Value = serde_json::from_str(error_str).ok()?;
    let Some(Value::Object(error)) = body.as_object()?.get("error") else {
        return None;
    };
    match error.get("code")? {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|value| value.trunc() as i64)),
        Value::String(code) => code.trim().replace('_', "").parse().ok(),
        Value::Bool(flag) => Some(i64::from(*flag)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::mapping;
    use super::*;

    const ORDERED: &[Rule] = &[
        Rule::new(
            |mapping| mapping.error_str.contains("first"),
            PublicError::NotFound,
        ),
        Rule::new(|_| true, PublicError::ApiConnection),
    ];

    #[rstest::rstest]
    #[case::earlier_rule_wins("first and second", PublicError::NotFound)]
    #[case::later_rule_when_the_earlier_does_not_apply("second", PublicError::ApiConnection)]
    fn the_first_applicable_rule_decides(#[case] text: &str, #[case] expected: PublicError) {
        let rule = first_match(ORDERED, &mapping(Some(400), text));
        assert_eq!(rule.map(|rule| rule.error), Some(expected));
    }

    #[test]
    fn no_applicable_rule_leaves_the_failure_to_the_caller() {
        assert!(first_match(&ORDERED[..1], &mapping(Some(400), "second")).is_none());
    }

    #[rstest::rstest]
    #[case::standalone_429_with_429_status("got 429 back", Some(429), true)]
    #[case::standalone_429_with_other_status("got 429 back", Some(400), false)]
    #[case::standalone_429_with_unknown_status("got 429 back", None, true)]
    #[case::embedded_429("token4290", Some(429), false)]
    #[case::phrase_spaced("Rate Limit reached", None, true)]
    #[case::phrase_underscored("rate_limit", None, true)]
    #[case::phrase_hyphenated("rate-limit", None, true)]
    #[case::service_tier("Service tier capacity exceeded", None, true)]
    #[case::unrelated("rejected", Some(429), false)]
    fn rate_limit_detection(
        #[case] text: &str,
        #[case] status: Option<u16>,
        #[case] expected: bool,
    ) {
        assert_eq!(is_rate_limit(text, status), expected);
    }

    #[rstest::rstest]
    #[case::exceed_context_limit("Exceed context limit", true)]
    #[case::maximum_context_length("This model's maximum context length is 10", true)]
    #[case::string_too_long("string too long. Expected a string with maximum length 5", true)]
    #[case::maximum_context_limit("the model's maximum context limit", true)]
    #[case::longer_than_context("prompt is longer than the model's context length", true)]
    #[case::configured_limit("input tokens exceed the configured limit", true)]
    #[case::max_new_tokens("`inputs` tokens + `max_new_tokens` must be <= 10", true)]
    #[case::available_context("exceeds the available context size", true)]
    #[case::maximum_tokens("exceeds the maximum number of tokens allowed", true)]
    #[case::current_and_limit("current length is 9 while limit is 8", true)]
    #[case::current_without_limit("current length is 9", false)]
    #[case::maximum_input_tokens("maximum input length is 8 tokens", true)]
    #[case::maximum_input_without_tokens("maximum input length is 8", false)]
    #[case::string_above_max_length_wins("string_above_max_length exceed context limit", false)]
    #[case::user_field_is_not_context(
        "invalid 'user': string too long. expected a string with maximum length",
        false
    )]
    #[case::unrelated("rejected", false)]
    fn context_window_detection(#[case] text: &str, #[case] expected: bool) {
        assert_eq!(is_context_window_exceeded(text), expected);
    }

    #[rstest::rstest]
    #[case::integer(r#"{"error": {"code": 429}}"#, Some(429))]
    #[case::float(r#"{"error": {"code": 429.9}}"#, Some(429))]
    #[case::string(r#"{"error": {"code": " 4_29 "}}"#, Some(429))]
    #[case::boolean(r#"{"error": {"code": true}}"#, Some(1))]
    #[case::unparseable_string(r#"{"error": {"code": "slow"}}"#, None)]
    #[case::null(r#"{"error": {"code": null}}"#, None)]
    #[case::no_code(r#"{"error": {}}"#, None)]
    #[case::error_not_an_object(r#"{"error": "429"}"#, None)]
    #[case::no_error(r#"{"code": 429}"#, None)]
    #[case::not_an_object("[429]", None)]
    #[case::not_json("429", None)]
    fn body_error_code_reads_the_nested_code(#[case] body: &str, #[case] expected: Option<i64>) {
        assert_eq!(body_error_code(body), expected);
    }
}
