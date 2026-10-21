use litellm_http::request::{header_values, with_default_headers, with_header};
use rstest::{fixture, rstest};

fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

#[rstest]
#[case::nothing_forwarded(
    &[],
    &[("x-version", "1"), ("content-type", "application/json")],
    &[("x-version", "1"), ("content-type", "application/json")],
)]
#[case::forwarded_header_wins_in_any_case(
    &[("X-Version", "custom"), ("x-api-key", "k")],
    &[("x-version", "1"), ("content-type", "application/json")],
    &[("X-Version", "custom"), ("x-api-key", "k"), ("content-type", "application/json")],
)]
#[case::no_defaults(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
fn default_headers_fill_only_missing_names(
    #[case] forwarded: &[(&str, &str)],
    #[case] defaults: &[(&str, &str)],
    #[case] expected: &[(&str, &str)],
) {
    assert_eq!(
        with_default_headers(headers(forwarded), defaults),
        headers(expected)
    );
}

#[fixture]
fn forwarded() -> Vec<(String, String)> {
    headers(&[
        ("X-Mode", "first"),
        ("x-trace", "trace"),
        ("x-MODE", "second"),
    ])
}

#[rstest]
#[case::matching_name("X-MODE", vec!["first", "second"])]
#[case::missing_name("missing", vec![])]
fn header_values_preserve_all_matching_values_in_order(
    forwarded: Vec<(String, String)>,
    #[case] name: &str,
    #[case] expected: Vec<&str>,
) {
    assert_eq!(
        header_values(&forwarded, name).collect::<Vec<_>>(),
        expected
    );
}

#[rstest]
#[case::replace("X-MODE", &[("x-trace", "trace"), ("x-mode", "new")])]
#[case::insert("X-New", &[("X-Mode", "first"), ("x-trace", "trace"), ("x-MODE", "second"), ("x-new", "new")])]
fn with_header_replaces_every_matching_name_and_preserves_other_headers(
    forwarded: Vec<(String, String)>,
    #[case] name: &str,
    #[case] expected: &[(&str, &str)],
) {
    assert_eq!(
        with_header(forwarded, name, "new".into()),
        headers(expected)
    );
}
