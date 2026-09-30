use litellm_traces::Connection;
use rstest::rstest;

#[rstest]
#[case::http("http://localhost:8123", true)]
#[case::https("https://localhost:8443", true)]
#[case::tcp("tcp://localhost:9000", false)]
#[case::missing_host("http://", false)]
fn accepts_only_clickhouse_http_urls(#[case] value: &str, #[case] expected: bool) {
    assert_eq!(Connection::parse(value).is_ok(), expected);
}
