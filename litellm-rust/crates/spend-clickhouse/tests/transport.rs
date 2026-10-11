use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_spend_clickhouse::{Config, Error, ensure_schema, insert_rows};
use rstest::rstest;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, query_param},
};

#[rstest]
#[tokio::test]
async fn spend_insert_controls_target_and_retry_settings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("Content-Encoding", "gzip"))
        .and(query_param(
            "query",
            "INSERT INTO `spend_test`.spend_logs FORMAT JSONEachRow",
        ))
        .and(query_param("async_insert", "1"))
        .and(query_param("async_insert_deduplicate", "1"))
        .and(query_param("wait_for_async_insert", "1"))
        .and(query_param("input_format_skip_unknown_fields", "0"))
        .and(query_param("date_time_input_format", "best_effort"))
        .respond_with(ResponseTemplate::new(200))
        .expect(2)
        .mount(&server)
        .await;
    let config = Config::new("spend_test".into(), &format!("{}?query=DROP&async_insert=0&wait_for_async_insert=0&input_format_skip_unknown_fields=1&database=wrong&readonly=1", server.uri()), 7).unwrap();
    let client = Client::no_redirect_for_test();
    let row = BTreeMap::from([
        ("request_id".into(), json!("request")),
        ("start_time".into(), json!(1234)),
    ]);
    insert_rows(
        &client,
        config.storage().writer(),
        "spend_test",
        vec![row.clone()],
    )
    .await
    .unwrap();
    insert_rows(&client, config.storage().writer(), "spend_test", vec![row])
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let token = |index: usize| {
        requests[index]
            .url
            .query_pairs()
            .find(|(key, _)| key == "insert_deduplication_token")
            .unwrap()
            .1
            .into_owned()
    };
    assert_eq!(token(0), token(1));
    assert!(
        !requests[0]
            .url
            .query_pairs()
            .any(|(key, _)| matches!(key.as_ref(), "database" | "readonly"))
    );
}

#[rstest]
#[case::denied(403)]
#[case::unavailable(503)]
#[tokio::test]
async fn spend_insert_preserves_storage_failure(#[case] status: u16) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(&server)
        .await;
    let config = Config::new("spend_test".into(), &server.uri(), 7).unwrap();
    let error = insert_rows(
        &Client::no_redirect_for_test(),
        config.storage().writer(),
        "spend_test",
        vec![BTreeMap::from([("request_id".into(), json!("request"))])],
    )
    .await
    .unwrap_err();
    assert!(
        matches!(error, Error::Storage(litellm_storage_clickhouse::Error::InsertFailed(code)) if code == status)
    );
}

#[rstest]
#[tokio::test]
async fn empty_spend_insert_does_not_contact_storage() {
    let server = MockServer::start().await;
    let config = Config::new("spend_test".into(), &server.uri(), 7).unwrap();
    insert_rows(
        &Client::no_redirect_for_test(),
        config.storage().writer(),
        "spend_test",
        vec![],
    )
    .await
    .unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn failed_schema_setup_is_reported() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403))
        .expect(1)
        .mount(&server)
        .await;
    let config = Config::new("spend_test".into(), &server.uri(), 7).unwrap();
    assert!(matches!(
        ensure_schema(&Client::no_redirect_for_test(), &config).await,
        Err(Error::Storage(
            litellm_storage_clickhouse::Error::SchemaFailed(403)
        ))
    ));
}

#[rstest]
#[case::database("bad-name", "http://localhost:8123", 7)]
#[case::retention("spend_test", "http://localhost:8123", 0)]
#[case::url("spend_test", "file:///tmp/storage", 7)]
fn invalid_configuration_is_rejected(
    #[case] database: &str,
    #[case] url: &str,
    #[case] retention: u32,
) {
    assert!(Config::new(database.into(), url, retention).is_err());
}
