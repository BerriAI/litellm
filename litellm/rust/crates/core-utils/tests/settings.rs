use litellm_core_utils::settings::resolve_non_empty;
use rstest::rstest;

#[rstest]
#[case::explicit_wins(Some("  explicit "), &["FIRST", "SECOND"], Some("explicit"))]
#[case::absent_falls_back(None, &["FIRST", "SECOND"], Some(" first "))]
#[case::blank_falls_back(Some(" \t "), &["BLANK", "SECOND"], Some("second"))]
#[case::skips_missing_and_blank(None, &["MISSING", "BLANK", "SECOND"], Some("second"))]
#[case::environment_order(None, &["SECOND", "FIRST"], Some("second"))]
#[case::missing(None, &["MISSING", "BLANK"], None)]
#[case::no_environment(None, &[], None)]
fn resolves_explicit_value_then_first_nonblank_environment_value(
    #[case] value: Option<&str>,
    #[case] names: &[&str],
    #[case] expected: Option<&str>,
) {
    let env = |name: &str| match name {
        "FIRST" => Some(" first ".to_string()),
        "SECOND" => Some("second".to_string()),
        "BLANK" => Some(" \t ".to_string()),
        _ => None,
    };
    assert_eq!(resolve_non_empty(value, &env, names).as_deref(), expected);
}

#[rstest]
fn explicit_value_does_not_read_the_environment() {
    let env = |_: &str| panic!("an explicit value must short-circuit environment lookup");
    assert_eq!(
        resolve_non_empty(Some("key"), &env, &["KEY"]).as_deref(),
        Some("key")
    );
}
