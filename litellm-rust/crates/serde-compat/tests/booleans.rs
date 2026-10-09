use std::collections::BTreeMap;

use litellm_serde_compat::{InvalidBoolean, parse_env_bool, parse_redis_bool, parse_str_bool};
use rstest::{fixture, rstest};

#[fixture]
fn tokens() -> BTreeMap<String, (String, Option<bool>)> {
    serde_json::from_str(include_str!("fixtures/env-booleans.json")).unwrap()
}

#[rstest]
#[case::true_token("true")]
#[case::false_token("false")]
#[case::one("one")]
#[case::zero("zero")]
#[case::yes("yes")]
#[case::no("no")]
#[case::on("on")]
#[case::off("off")]
#[case::t("t")]
#[case::f("f")]
#[case::y("y")]
#[case::n("n")]
#[case::mixed_true("mixed_true")]
#[case::mixed_false("mixed_false")]
#[case::upper_t("upper_t")]
#[case::upper_f("upper_f")]
#[case::upper_y("upper_y")]
#[case::upper_n("upper_n")]
#[case::padded_true("padded_true")]
#[case::padded_false("padded_false")]
#[case::python_controls("python_controls")]
#[case::unicode_whitespace("unicode_whitespace")]
#[case::empty("empty")]
#[case::blank("blank")]
#[case::whitespace_only("whitespace_only")]
#[case::garbage("garbage")]
#[case::two("two")]
#[case::zero_width_space("zero_width_space")]
#[case::null("null")]
#[case::embedded_whitespace("embedded_whitespace")]
fn environment_tokens_match_the_python_decoder(
    tokens: BTreeMap<String, (String, Option<bool>)>,
    #[case] name: &str,
) {
    let (input, expected) = &tokens[name];
    assert_eq!(parse_env_bool(input), expected.ok_or(InvalidBoolean));
}

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
#[case::one("1", true)]
#[case::true_token("TrUe", true)]
#[case::yes("YES", true)]
#[case::padded(" true ", false)]
#[case::on("on", false)]
#[case::short("t", false)]
#[case::false_token("false", false)]
#[case::empty("", false)]
fn redis_tokens_preserve_their_existing_contract(#[case] input: &str, #[case] expected: bool) {
    assert_eq!(parse_redis_bool(input), expected);
}
