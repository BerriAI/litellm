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

#[rstest]
#[case::writer_only(None, false)]
#[case::separate_reader(Some("http://localhost:8124"), true)]
fn storage_exports_writer_and_optional_reader(
    #[case] reader_url: Option<&str>,
    #[case] has_reader: bool,
) {
    let storage = Storage::new("litellm".to_owned(), "http://localhost:8123", reader_url)
        .expect("valid ClickHouse URLs");

    assert_eq!(storage.database(), "litellm");
    assert_eq!(storage.writer().url().host_str(), Some("localhost"));
    assert_eq!(storage.writer().url().port(), Some(8123));
    assert_eq!(storage.reader().is_some(), has_reader);
}

#[rstest]
#[case::empty("")]
#[case::injection("db; DROP DATABASE default")]
fn storage_rejects_invalid_database(#[case] database: &str) {
    assert!(Storage::new(database.to_owned(), "http://localhost:8123", None).is_err());
}
