use litellm_traces::{InvalidQuery, ReadQuery};
use rstest::rstest;

#[rstest]
#[case::list_traces("list_traces", ReadQuery::ListTraces)]
#[case::trace_agents("trace_agents", ReadQuery::TraceAgents)]
#[case::trace_spans("trace_spans", ReadQuery::TraceSpans)]
#[case::span_detail("span_detail", ReadQuery::SpanDetail)]
#[case::span_error("span_error", ReadQuery::SpanError)]
#[case::identity("trace_identity", ReadQuery::TraceIdentity)]
#[case::spend("spend_by_response_ids", ReadQuery::SpendByResponseIds)]
#[case::availability("availability", ReadQuery::Availability)]
#[case::agents("agents", ReadQuery::Agents)]
#[case::sample("sample", ReadQuery::Sample)]
#[case::content("content", ReadQuery::Content)]
#[case::evidence("evidence", ReadQuery::Evidence)]
#[case::feedback_target("feedback_target", ReadQuery::FeedbackTarget)]
#[case::feedback("feedback", ReadQuery::Feedback)]
#[case::feedback_summary("feedback_summary", ReadQuery::FeedbackSummary)]
fn names_select_the_public_query(#[case] name: &str, #[case] query: ReadQuery) {
    assert_eq!(ReadQuery::parse(name).unwrap(), query);
    assert_eq!(query.as_ref(), name);
    assert_eq!(query.to_string(), name);
}

#[rstest]
#[case::unknown("unknown")]
#[case::case_sensitive("List_Traces")]
#[case::whitespace(" list_traces")]
#[case::empty("")]
fn invalid_names_preserve_the_public_error(#[case] name: &str) {
    let error = ReadQuery::parse(name).unwrap_err();
    assert!(matches!(error, InvalidQuery));
    assert_eq!(error.to_string(), "unknown ClickHouse read query");
}

#[path = "query/named.rs"]
mod named;
