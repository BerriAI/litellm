//! `yaml.safe_load` parity for what a config uses. Each expected value is what PyYAML 6.0.3 printed
//! for the text as `repr()`, or `error` where it raised:
//! `uv run --no-project --with pyyaml==6.0.3 python -I -c 'import yaml; print(repr(yaml.safe_load(TEXT)))'`

use litellm_python_compat::{Value, repr::repr};
use rstest::rstest;
use serde::Deserialize;

fn loaded(text: &str) -> String {
    litellm_yaml::from_str(text).map_or_else(
        |_| "error".to_owned(),
        |value| repr(&Value::deserialize(&value).expect("a loaded document is plain data")),
    )
}

#[rstest]
#[case::yes("v: yes", "{'v': True}")]
#[case::capital_yes("v: Yes", "{'v': True}")]
#[case::upper_no("v: NO", "{'v': False}")]
#[case::on("v: on", "{'v': True}")]
#[case::off("v: Off", "{'v': False}")]
#[case::y_is_text("v: y", "{'v': 'y'}")]
#[case::n_is_text("v: n", "{'v': 'n'}")]
#[case::mixed_case_true_is_text("v: tRUE", "{'v': 'tRUE'}")]
#[case::tilde("v: ~", "{'v': None}")]
#[case::null_word("v: Null", "{'v': None}")]
#[case::empty("v:", "{'v': None}")]
#[case::upper_null("v: NULL", "{'v': None}")]
#[case::octal("v: 0755", "{'v': 493}")]
#[case::zero_o_is_text("v: 0o17", "{'v': '0o17'}")]
#[case::hex("v: 0x1F", "{'v': 31}")]
#[case::binary("v: 0b101", "{'v': 5}")]
#[case::underscores("v: 1_000", "{'v': 1000}")]
#[case::plus("v: +1", "{'v': 1}")]
#[case::negative_zero("v: -0", "{'v': 0}")]
#[case::zero("v: 0", "{'v': 0}")]
#[case::base_60("v: 1:30", "{'v': 90}")]
#[case::base_60_three("v: 190:20:30", "{'v': 685230}")]
#[case::base_60_float("v: 1:30.5", "{'v': 90.5}")]
#[case::base_60_bad_group_is_text("v: 1:60", "{'v': '1:60'}")]
#[case::leading_zero_nine_is_text("v: 09", "{'v': '09'}")]
#[case::octal_underscore("v: 0_7", "{'v': 7}")]
#[case::exponent_without_dot_is_text("v: 1e3", "{'v': '1e3'}")]
#[case::exponent_without_sign_is_text("v: 1.0e3", "{'v': '1.0e3'}")]
#[case::signed_exponent("v: 1.0e+3", "{'v': 1000.0}")]
#[case::trailing_dot("v: 1.", "{'v': 1.0}")]
#[case::leading_dot("v: .5", "{'v': 0.5}")]
#[case::signed_leading_dot_is_text("v: -.5", "{'v': '-.5'}")]
#[case::dot_exponent("v: .5e-1", "{'v': 0.05}")]
#[case::inf("v: .inf", "{'v': inf}")]
#[case::negative_inf("v: -.Inf", "{'v': -inf}")]
#[case::nan("v: .NaN", "{'v': nan}")]
#[case::signed_nan_is_text("v: -.nan", "{'v': '-.nan'}")]
#[case::float_underscores("v: 1.5_0", "{'v': 1.5}")]
#[case::negative_float("v: -2.5", "{'v': -2.5}")]
#[case::lone_dot_is_text("v: .", "{'v': '.'}")]
#[case::i64_max("v: 9223372036854775807", "{'v': 9223372036854775807}")]
#[case::u64_max("v: 18446744073709551615", "{'v': 18446744073709551615}")]
#[case::date_short_parts_is_text("v: 2026-1-5", "{'v': '2026-1-5'}")]
#[case::api_version("v: 2024-02-15-preview", "{'v': '2024-02-15-preview'}")]
#[case::value_key("=: 1", "{'=': 1}")]
#[case::single_quoted_yes("v: 'yes'", "{'v': 'yes'}")]
#[case::double_quoted_octal("v: \"0755\"", "{'v': '0755'}")]
#[case::str_tag("v: !!str 0755", "{'v': '0755'}")]
#[case::int_tag_on_quoted("v: !!int '12'", "{'v': 12}")]
#[case::float_tag_on_int("v: !!float 1", "{'v': 1.0}")]
#[case::bool_tag("v: !!bool yes", "{'v': True}")]
#[case::bool_tag_unknown("v: !!bool maybe", "error")]
#[case::int_tag_on_text("v: !!int abc", "error")]
#[case::null_tag_on_text("v: !!null anything", "{'v': None}")]
#[case::non_specific_tag("v: ! 123", "{'v': 123}")]
#[case::non_specific_tag_quoted("v: ! '123'", "{'v': 123}")]
#[case::custom_tag("v: !custom 1", "error")]
#[case::python_tag("v: !!python/name:os.system", "error")]
#[case::seq_tag_on_scalar("v: !!seq x", "error")]
#[case::str_tag_on_mapping("v: !!str {a: 1}", "error")]
#[case::map_tag("v: !!map {a: 1}", "{'v': {'a': 1}}")]
#[case::verbatim_str_tag("v: !<tag:yaml.org,2002:str> 1", "{'v': '1'}")]
#[case::duplicate_key("a: 1\nb: 2\na: 3", "{'a': 3, 'b': 2}")]
#[case::merge(
    "b: &b {x: 1, y: 2}\nm: {<<: *b, y: 3}",
    "{'b': {'x': 1, 'y': 2}, 'm': {'x': 1, 'y': 3}}"
)]
#[case::merge_list_earlier_wins(
    "a: &a {k: a}\nb: &b {k: b, j: b}\nm: {<<: [*a, *b]}",
    "{'a': {'k': 'a'}, 'b': {'k': 'b', 'j': 'b'}, 'm': {'k': 'a', 'j': 'b'}}"
)]
#[case::own_key_before_merge_wins(
    "a: &a {k: a, j: a}\nm: {k: own, <<: *a}",
    "{'a': {'k': 'a', 'j': 'a'}, 'm': {'k': 'own', 'j': 'a'}}"
)]
#[case::merge_sequence_errors("a: &a [1]\nm: {<<: *a}", "error")]
#[case::later_merge_wins(
    "a: &a {k: 1}\nm: {<<: *a, <<: {k: 2, j: 2}}",
    "{'a': {'k': 1}, 'm': {'k': 2, 'j': 2}}"
)]
#[case::quoted_merge_key("m: {'<<': {a: 1}}", "{'m': {'<<': {'a': 1}}}")]
#[case::explicit_merge_tag("a: &a {k: 1}\nm: {!!merge x: *a}", "{'a': {'k': 1}, 'm': {'k': 1}}")]
#[case::nested_merge(
    "a: &a {k: 1}\nb: &b {<<: *a, j: 2}\nm: {<<: *b}",
    "{'a': {'k': 1}, 'b': {'k': 1, 'j': 2}, 'm': {'k': 1, 'j': 2}}"
)]
#[case::merge_as_value_errors("v: <<", "error")]
#[case::merge_scalar_errors("m: {<<: 1}", "error")]
#[case::multiple_documents("---\na: 1\n---\nb: 2", "error")]
#[case::yaml_directive("%YAML 1.1\n---\na: 1", "{'a': 1}")]
#[case::explicit_end("a: 1\n...", "{'a': 1}")]
#[case::tab_indent("a: x\n\tb: y", "error")]
#[case::flow_pair_in_sequence("a: [b: c]", "{'a': [{'b': 'c'}]}")]
#[case::sequence_key_errors("a: {? [1]: 2}", "error")]
#[case::empty_str_tag("a: !!str\nb: 1", "{'a': '', 'b': 1}")]
#[case::escapes("a: 'it''s'\nb: \"\\/\\t\"", "{'a': \"it's\", 'b': '/\\t'}")]
#[case::bom("\u{feff}a: 1", "{'a': 1}")]
#[case::alias("a: &x 1\nb: *x", "{'a': 1, 'b': 1}")]
#[case::comment("key with spaces: v # comment", "{'key with spaces': 'v'}")]
#[case::folded("a: >-\n  folded\n  text", "{'a': 'folded text'}")]
#[case::literal("a: |\n  line\n  next\n", "{'a': 'line\\nnext\\n'}")]
#[case::empty_document("", "None")]
#[case::comment_only("# nothing", "None")]
#[case::top_level_list("- 1\n- yes", "[1, True]")]
#[case::unknown_alias("a: *missing", "error")]
#[case::recursive_alias("a: &a [*a]", "error")]
fn loads_like_pyyaml_safe_load(#[case] text: &str, #[case] expected: &str) {
    assert_eq!(loaded(text), expected);
}

/// Where this port differs from PyYAML on purpose, see `AGENTS.md`. Each case names what PyYAML
/// returns.
#[rstest]
/// PyYAML: `{True: 3, None: 4, 1.5: 5}`, since `1 == True` makes the two keys one. A `Value`
/// mapping compares keys by kind, so the port keeps both.
#[case::bool_and_int_keys_stay_apart(
    "yes: 1\non: 2\n1: 3\nnull: 4\n1.5: 5",
    "{True: 2, 1: 3, None: 4, 1.5: 5}"
)]
/// PyYAML: `{'v': 18446744073709551616}`. A `Value` number holds 64 bits.
#[case::integer_beyond_u64("v: 18446744073709551616", "error")]
/// PyYAML: `{'v': datetime.date(2026, 11, 16)}`, which no config field accepts as a string.
#[case::date_stays_text("v: 2026-11-16", "{'v': '2026-11-16'}")]
/// PyYAML: `{'v': datetime.datetime(2026, 11, 16, 10, 0, tzinfo=datetime.timezone.utc)}`.
#[case::datetime_stays_text("v: 2026-11-16T10:00:00Z", "{'v': '2026-11-16T10:00:00Z'}")]
/// PyYAML raises, since February has no 30th.
#[case::invalid_date_stays_text("v: 2026-02-30", "{'v': '2026-02-30'}")]
/// PyYAML raises, having no constructor for the `value` tag a bare `=` resolves to.
#[case::equals_sign_stays_text("v: =", "{'v': '='}")]
/// PyYAML: `{'v': {'a'}}`.
#[case::set_is_refused("v: !!set {a}", "error")]
/// PyYAML: `{'v': [('a', 1), ('b', 2)]}`, and the same shape for `!!pairs`.
#[case::omap_is_refused("v: !!omap [a: 1, b: 2]", "error")]
/// PyYAML: `{'v': b'hi'}`.
#[case::binary_is_refused("v: !!binary aGk=", "error")]
/// PyYAML: `{'v': datetime.date(2026, 11, 16)}`.
#[case::timestamp_tag_is_refused("v: !!timestamp 2026-11-16", "error")]
fn differs_from_pyyaml_on_purpose(#[case] text: &str, #[case] expected: &str) {
    assert_eq!(loaded(text), expected);
}
