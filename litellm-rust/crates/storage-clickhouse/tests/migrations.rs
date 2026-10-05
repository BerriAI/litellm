use std::time::Duration;

use litellm_http::Client;
use litellm_migrate::{ChangedMigration, Migration};
use litellm_storage_clickhouse::{Connection, Error, apply_migrations};
use rstest::rstest;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string, method},
};

const FIRST_CHECKSUM: &str = "e004ebd5b5532a4b85984a62f8ad48a81aa3460c1ca07701f386135d72cdecf5";
const SECOND_CHECKSUM: &str = "ebbb5b332060a3ede7047bd7528883b0a560e729848f9feb8ff742145e909b01";
const MIGRATIONS: [Migration; 2] = [
    Migration {
        version: 1,
        description: "first",
        sql: "SELECT 1",
        checksum: FIRST_CHECKSUM,
    },
    Migration {
        version: 2,
        description: "second",
        sql: "SELECT 2",
        checksum: SECOND_CHECKSUM,
    },
];

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
#[tokio::test]
async fn applies_only_pending_migrations_and_records_their_checksums() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(format!(
            "{{\"version\":1,\"checksum\":\"{FIRST_CHECKSUM}\"}}\n"
        )))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
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
            "SELECT 2".to_owned(),
            "INSERT INTO `trace_test`.schema_migrations (version, description, checksum) VALUES (2, 'second', 'ebbb5b332060a3ede7047bd7528883b0a560e729848f9feb8ff742145e909b01')".to_owned(),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn rejects_applied_migration_with_a_changed_checksum() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "{\"version\":1,\"checksum\":\"d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57\"}\n",
        ))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
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
async fn rejects_invalid_database_before_sending_requests() {
    let server = MockServer::start().await;
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
#[tokio::test]
async fn failed_migration_statement_is_not_recorded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_string(
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow",
        ))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_string("SELECT 1"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
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

    assert!(matches!(result, Err(Error::SchemaFailed(500))));
    assert_eq!(
        received_bodies(&server).await,
        [
            "CREATE DATABASE IF NOT EXISTS `trace_test`".to_owned(),
            create_ledger_statement("trace_test"),
            "SELECT version, checksum FROM `trace_test`.schema_migrations FORMAT JSONEachRow"
                .to_owned(),
            "SELECT 1".to_owned(),
        ]
    );
}
