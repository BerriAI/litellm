use std::time::Duration;

use litellm_http::Client;
use litellm_migrate::{ChangedMigration, Migration};
use litellm_storage_clickhouse::{
    Connection, Error, READ_LIMITS, apply_migrations, execute_statement,
};
use rstest::{fixture, rstest};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string, method, query_param},
};

const FIRST_CHECKSUM: &str = "a408b4b0f9fd588711feef16b8dfe6e8aa3a322a51f410ae6d05b142b5cd5704";
const SECOND_CHECKSUM: &str = "352216a05ec02310c33edf7ab06b18ac241f287bb163b6f2a825626e90c223e1";
const MIGRATIONS: [Migration; 2] = [
    Migration {
        version: 1,
        description: "first",
        sql: "CREATE TABLE IF NOT EXISTS `trace_test`.marker (id UInt64) ENGINE=MergeTree ORDER BY id",
        checksum: FIRST_CHECKSUM,
    },
    Migration {
        version: 2,
        description: "second",
        sql: "ALTER TABLE `trace_test`.marker ADD COLUMN IF NOT EXISTS name String",
        checksum: SECOND_CHECKSUM,
    },
];

#[fixture]
async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .with_priority(10)
        .mount(&server)
        .await;
    server
}

fn create_ledger_statement(database: &str) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS `{database}`.schema_migrations \
         (version UInt64, description String, checksum String, applied_at DateTime DEFAULT now()) \
         ENGINE = MergeTree ORDER BY version"
    )
}

async fn received_bodies(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("requests were recorded")
        .iter()
        .map(|request| String::from_utf8(request.body.clone()).expect("request body is UTF-8"))
        .collect()
}

#[rstest]
#[case::numeric("1")]
#[case::quoted("\"1\"")]
#[tokio::test]
async fn applies_only_pending_migrations_and_records_their_checksums(
    #[future(awt)] server: MockServer,
    #[case] version: &str,
) {
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            "{{\"version\":{version},\"checksum\":\"{FIRST_CHECKSUM}\"}}\n"
        )))
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).expect("valid URL");

    apply_migrations(
        &client,
        &connection,
        "trace_test",
        &MIGRATIONS,
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await
    .expect("pending migration applies");

    assert_eq!(
        received_bodies(&server).await,
        [
            "CREATE DATABASE IF NOT EXISTS `trace_test`".to_owned(),
            create_ledger_statement("trace_test"),
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow".to_owned(),
            "ALTER TABLE `trace_test`.marker ADD COLUMN IF NOT EXISTS name String".to_owned(),
            "INSERT INTO `trace_test`.schema_migrations (version, description, checksum) VALUES (2, 'second', '352216a05ec02310c33edf7ab06b18ac241f287bb163b6f2a825626e90c223e1')".to_owned(),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn rejects_applied_migration_with_a_changed_checksum(#[future(awt)] server: MockServer) {
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "{\"version\":1,\"checksum\":\"d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57\"}\n",
        ))
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).expect("valid URL");

    let result = apply_migrations(
        &client,
        &connection,
        "trace_test",
        &MIGRATIONS,
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await;

    assert!(matches!(
        result,
        Err(Error::Migration(ChangedMigration { version: 1 }))
    ));
    assert_eq!(
        received_bodies(&server).await,
        [
            "CREATE DATABASE IF NOT EXISTS `trace_test`".to_owned(),
            create_ledger_statement("trace_test"),
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow"
                .to_owned(),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn rejects_invalid_database_before_sending_requests(#[future(awt)] server: MockServer) {
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).expect("valid URL");

    let result = apply_migrations(
        &client,
        &connection,
        "trace_test; DROP DATABASE default",
        &MIGRATIONS,
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await;

    assert!(matches!(result, Err(Error::InvalidSchema)));
    assert!(received_bodies(&server).await.is_empty());
}

#[rstest]
#[case::http_error(500, "")]
#[case::json_error(200, "{\"exception\":\"failed\"}\n")]
#[case::streamed_error(200, "{\"row\":1}\n{\"exception\":\"failed\"}\n")]
#[case::text_error(200, "Code: 395. DB::Exception: failed\n")]
#[case::framed_error(
    200,
    "\r\n__exception__\r\ntag\r\nfailed\r\n6 tag\r\n__exception__\r\n"
)]
#[case::unexpected_output(200, "1\n")]
#[tokio::test]
async fn failed_migration_statement_is_not_recorded(
    #[future(awt)] server: MockServer,
    #[case] status: u16,
    #[case] body: &str,
) {
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_string("CREATE TABLE IF NOT EXISTS `trace_test`.marker (id UInt64) ENGINE=MergeTree ORDER BY id"))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).expect("valid URL");

    let result = apply_migrations(
        &client,
        &connection,
        "trace_test",
        &MIGRATIONS[..1],
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await;

    match status {
        500 => assert!(matches!(result, Err(Error::SchemaFailed(500)))),
        _ => assert!(matches!(result, Err(Error::InvalidResponse))),
    }
    assert_eq!(
        received_bodies(&server).await,
        [
            "CREATE DATABASE IF NOT EXISTS `trace_test`".to_owned(),
            create_ledger_statement("trace_test"),
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow"
                .to_owned(),
            "CREATE TABLE IF NOT EXISTS `trace_test`.marker (id UInt64) ENGINE=MergeTree ORDER BY id".to_owned(),
        ]
    );
}

#[rstest]
#[case::missing_version("{\"checksum\":\"first\"}")]
#[case::invalid_version("{\"version\":\"invalid\",\"checksum\":\"first\"}")]
#[case::negative_version("{\"version\":-1,\"checksum\":\"first\"}")]
#[case::exception("{\"exception\":\"failed\"}")]
#[tokio::test]
async fn invalid_ledger_stops_before_migrations(
    #[future(awt)] server: MockServer,
    #[case] body: &str,
) {
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let result = apply_migrations(
        &Client::no_redirect_for_test(),
        &Connection::parse(&server.uri()).expect("valid URL"),
        "trace_test",
        &MIGRATIONS,
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(result, Err(Error::InvalidResponse)));
    assert_eq!(received_bodies(&server).await.len(), 3);
}

#[rstest]
#[tokio::test]
async fn schema_requests_override_unsafe_connection_settings(#[future(awt)] server: MockServer) {
    Mock::given(method("POST"))
        .and(query_param("wait_end_of_query", "1"))
        .and(query_param("send_progress_in_http_headers", "0"))
        .and(query_param("async_insert", "0"))
        .and(query_param("custom_setting", "preserved"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    let url = format!(
        "{}/?wait_end_of_query=0&send_progress_in_http_headers=1&async_insert=1&custom_setting=preserved",
        server.uri()
    );
    execute_statement(
        &Client::no_redirect_for_test(),
        &Connection::parse(&url).expect("valid URL"),
        "CREATE DATABASE IF NOT EXISTS trace_test",
        Duration::from_secs(5),
    )
    .await
    .expect("schema execution succeeds");
    let requests = server.received_requests().await.expect("requests recorded");
    for name in [
        "wait_end_of_query",
        "send_progress_in_http_headers",
        "async_insert",
    ] {
        assert_eq!(
            requests[0]
                .url
                .query_pairs()
                .filter(|(key, _)| key == name)
                .count(),
            1
        );
    }
}

#[rstest]
#[tokio::test]
async fn oversized_ledger_stops_before_migrations(#[future(awt)] server: MockServer) {
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(" ".repeat(READ_LIMITS.response_bytes + 1)),
        )
        .mount(&server)
        .await;
    let result = apply_migrations(
        &Client::no_redirect_for_test(),
        &Connection::parse(&server.uri()).expect("valid URL"),
        "trace_test",
        &MIGRATIONS,
        |migration| migration.sql.to_owned(),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(result, Err(Error::ResponseTooLarge)));
    assert_eq!(received_bodies(&server).await.len(), 3);
}
