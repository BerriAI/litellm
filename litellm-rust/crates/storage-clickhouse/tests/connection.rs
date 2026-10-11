use litellm_storage_clickhouse::{Connection, Storage};
use rstest::rstest;

#[rstest]
#[case::http("http://localhost:8123", true)]
#[case::https("https://localhost:8443", true)]
#[case::tcp("tcp://localhost:9000", false)]
#[case::missing_host("http://", false)]
fn accepts_only_clickhouse_http_urls(#[case] value: &str, #[case] expected: bool) {
    assert_eq!(Connection::parse(value).is_ok(), expected);
}

#[test]
fn storage_uses_one_url_for_writes_and_bounded_reads() {
    let storage =
        Storage::new("litellm".to_owned(), "http://localhost:8123").expect("valid ClickHouse URLs");

    assert_eq!(storage.database(), "litellm");
    assert_eq!(storage.writer().url().host_str(), Some("localhost"));
    assert_eq!(storage.writer().url().port(), Some(8123));
    assert_eq!(storage.reader().url().port(), Some(8123));
    assert_eq!(
        storage
            .reader()
            .url()
            .query_pairs()
            .find(|(key, _)| key == "database")
            .unwrap()
            .1,
        "litellm"
    );
}

#[rstest]
#[case::empty("")]
#[case::injection("db; DROP DATABASE default")]
fn storage_rejects_invalid_database(#[case] database: &str) {
    assert!(Storage::new(database.to_owned(), "http://localhost:8123").is_err());
}
