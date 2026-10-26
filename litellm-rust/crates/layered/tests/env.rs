use litellm_core_utils::coercion;
use std::{path::PathBuf, time::Duration};

use litellm_layered::{Env, Error, Layered, Lookup, read_env};
use rstest::rstest;
use serde_json::{Value, json};

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is built")]
struct Settings {
    #[layer(env = ["FIRST_NAME", "SECOND_NAME"], skip_blank)]
    name: String,
    #[layer(env = "SWITCH", read = coercion::enabled)]
    switch: bool,
    #[layer(env = "COUNT")]
    count: u32,
    #[layer(env = "PORT", read = coercion::required::<u16>)]
    port: u16,
    #[layer(env = "IDLE", read = coercion::seconds)]
    idle: Duration,
    #[layer(from_env = window_from_env)]
    window: (u32, u32),
    #[layer(litellm_param = "label", env = "LABEL")]
    label: String,
    untouched_by_env: bool,
}

fn window_from_env(env: &(impl Lookup + ?Sized)) -> Result<Option<(u32, u32)>, Error> {
    let start = read_env(env, &["WINDOW_START"], false, coercion::required::<u32>)?;
    let end = read_env(env, &["WINDOW_END"], false, coercion::required::<u32>)?;
    Ok(start.zip(end))
}

fn params(value: Value) -> serde_json::Map<String, Value> {
    serde_json::from_value(value).unwrap()
}

#[rstest]
fn each_field_reads_its_env_names_in_order() {
    let partial = SettingsPartial::from_env(&Env::of(&[
        ("SECOND_NAME", "second"),
        ("SWITCH", "True"),
        ("COUNT", " 7 "),
        ("PORT", "8080"),
        ("IDLE", "45"),
        ("WINDOW_START", "1"),
        ("WINDOW_END", "2"),
    ]))
    .unwrap();
    assert_eq!(
        partial,
        SettingsPartial {
            name: Some("second".into()),
            switch: Some(true),
            count: Some(7),
            port: Some(8080),
            idle: Some(Duration::from_secs(45)),
            window: Some((1, 2)),
            label: None,
            untouched_by_env: None,
        }
    );
}

#[rstest]
#[case::empty("")]
#[case::whitespace("  ")]
fn blank_values_fall_through_to_the_next_name(#[case] blank: &str) {
    let partial = SettingsPartial::from_env(&Env::of(&[
        ("FIRST_NAME", blank),
        ("SECOND_NAME", "second"),
    ]))
    .unwrap();
    assert_eq!(partial.name.as_deref(), Some("second"));
}

#[rstest]
fn text_fields_preserve_blank_values_by_default() {
    assert_eq!(
        SettingsPartial::from_env(&Env::of(&[("LABEL", "")]))
            .unwrap()
            .label
            .as_deref(),
        Some("")
    );
    assert_eq!(
        SettingsPartial::from_litellm_params(&params(json!({"label": " "})))
            .label
            .as_deref(),
        Some(" ")
    );
}

#[rstest]
fn a_field_without_env_never_comes_from_env() {
    let partial = SettingsPartial::from_env(&Env::of(&[("UNTOUCHED_BY_ENV", "true")])).unwrap();
    assert_eq!(partial, SettingsPartial::default());
}

#[rstest]
#[case::switch_off("SWITCH", "false")]
#[case::switch_one("SWITCH", "1")]
#[case::lax_count("COUNT", "many")]
fn lax_values_that_do_not_read_are_unset(#[case] name: &str, #[case] value: &str) {
    assert_eq!(
        SettingsPartial::from_env(&Env::of(&[(name, value)])).unwrap(),
        SettingsPartial::default()
    );
}

#[rstest]
#[case::port("PORT", "abc")]
#[case::port_overflow("PORT", "65536")]
#[case::from_env_field("WINDOW_START", "soon")]
fn strict_values_that_do_not_read_fail_naming_the_variable(
    #[case] name: &'static str,
    #[case] value: &str,
) {
    let error = SettingsPartial::from_env(&Env::of(&[(name, value)])).unwrap_err();
    assert_eq!(error.name, name);
    assert!(error.to_string().contains(name));
}

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is built")]
struct Values {
    #[layer(env = "TEXT")]
    text: String,
    #[layer(env = "PATH")]
    path: PathBuf,
    #[layer(env = "BOOL")]
    boolean: bool,
}

#[rstest]
#[case::empty("")]
#[case::whitespace(" \t ")]
#[case::padded(" value ")]
fn inferred_strings_and_paths_preserve_the_input(#[case] value: &str) {
    let partial = ValuesPartial::from_env(&Env::of(&[("TEXT", value), ("PATH", value)])).unwrap();
    assert_eq!(partial.text.as_deref(), Some(value));
    assert_eq!(partial.path, Some(PathBuf::from(value)));
}

#[rstest]
#[case::true_value("True", Some(true))]
#[case::false_value(" FALSE ", Some(false))]
#[case::python_whitespace("\u{1c}false\u{1f}", Some(false))]
#[case::invalid("yes", None)]
#[case::empty("", None)]
fn inferred_booleans_keep_false_distinct_from_unset(
    #[case] value: &str,
    #[case] expected: Option<bool>,
) {
    let partial = ValuesPartial::from_env(&Env::of(&[("BOOL", value)])).unwrap();
    assert_eq!(partial.boolean, expected);
}
