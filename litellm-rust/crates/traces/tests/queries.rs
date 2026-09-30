use litellm_traces::{Connection, Error, ListQuery, Query};
use rstest::rstest;

fn list(start_ms: i64, end_ms: i64, limit: u8) -> Query {
    Query::List(ListQuery {
        start_ms,
        end_ms,
        service: None,
        status: None,
        search: None,
        cursor: None,
        limit,
    })
}

#[rstest]
#[case::reversed_window(list(2, 1, 50))]
#[case::empty_window(list(1, 1, 50))]
#[case::zero_limit(list(1, 2, 0))]
#[case::large_limit(list(1, 2, 101))]
#[case::empty_trace(Query::Trace { trace_id: String::new() })]
#[case::empty_span(Query::Span { trace_id: "trace".into(), span_id: String::new() })]
fn rejects_invalid_queries(#[case] request: Query) {
    assert!(request.validate().is_err());
}

#[rstest]
#[case::http("http://localhost:8123", true)]
#[case::https("https://localhost:8443", true)]
#[case::tcp("tcp://localhost:9000", false)]
#[case::missing_host("http://", false)]
fn accepts_only_clickhouse_http_urls(#[case] value: &str, #[case] expected: bool) {
    assert_eq!(Connection::parse(value).is_ok(), expected);
}

#[tokio::test]
async fn schema_pending_is_explicit_after_valid_request() {
    let connection = Connection::parse("http://localhost:8123").expect("valid URL");
    let result = litellm_traces::query(connection, list(1, 2, 50)).await;
    assert!(matches!(result, Err(Error::SchemaPending)));
}
