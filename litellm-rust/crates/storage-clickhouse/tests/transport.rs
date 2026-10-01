use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_storage_clickhouse::{Connection, Error, execute_read, insert_encoded_rows};
use rstest::rstest;

#[rstest]
#[case::invalid_database("db; DROP DATABASE default", "spend_logs", true)]
#[case::invalid_table("litellm", "spend_logs; DROP TABLE otel_traces", false)]
#[tokio::test]
async fn insert_rejects_invalid_identifiers(
    #[case] database: &str,
    #[case] table: &str,
    #[case] invalid_database: bool,
) {
    let client = Client::no_redirect_for_test();
    let connection = Connection::writer("http://localhost:8123").expect("valid URL");
    let result = insert_encoded_rows(&client, &connection, database, table, "token", "{}").await;

    assert!(matches!(&result, Err(Error::InvalidSchema)) == invalid_database);
    assert!(matches!(&result, Err(Error::InvalidTable)) == !invalid_database);
}

#[rstest]
#[tokio::test]
async fn read_rejects_empty_sql() {
    let client = Client::no_redirect_for_test();
    let connection = Connection::reader("http://localhost:8123", "litellm").expect("valid URL");

    assert!(matches!(
        execute_read(&client, &connection, "  ", &BTreeMap::new()).await,
        Err(Error::EmptySql)
    ));
}
