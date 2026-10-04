use litellm_traces::{InvalidQuery, ReadQuery};
use rstest::rstest;

#[rstest]
#[case::availability("availability", ReadQuery::Availability)]
#[case::agents("agents", ReadQuery::Agents)]
#[case::sample("sample", ReadQuery::Sample)]
#[case::content("content", ReadQuery::Content)]
#[case::evidence("evidence", ReadQuery::Evidence)]
fn names_select_the_public_query(#[case] name: &str, #[case] query: ReadQuery) {
    assert_eq!(ReadQuery::parse(name).unwrap(), query);
    assert_eq!(query.as_ref(), name);
    assert_eq!(query.to_string(), name);
}

#[rstest]
#[case::unknown("unknown")]
#[case::case_sensitive("Sample")]
#[case::whitespace(" sample")]
#[case::empty("")]
fn invalid_names_preserve_the_public_error(#[case] name: &str) {
    let error = ReadQuery::parse(name).unwrap_err();
    assert!(matches!(error, InvalidQuery));
    assert_eq!(error.to_string(), "unknown ClickHouse read query");
}
