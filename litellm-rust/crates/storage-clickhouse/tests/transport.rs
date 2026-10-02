use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_storage_clickhouse::{Connection, Error, Query, execute_read, insert_encoded_rows};
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

#[derive(serde::Serialize)]
struct QueryParams {
    signed: i64,
    unsigned: u64,
    float: f64,
    text: String,
    strings: Vec<String>,
}

#[derive(Debug, serde::Deserialize, PartialEq)]
struct QueryRow {
    answer: String,
}

struct TypedQuery;

impl litellm_storage_clickhouse::Query for TypedQuery {
    type Params = QueryParams;
    type Row = QueryRow;
    const SQL: &'static str = "SELECT typed_parameters";
}

#[rstest]
#[case::valid(
    r#"{"meta":[],"data":[{"answer":"ok"}],"rows":1,"statistics":{"elapsed":0.1}}"#,
    true
)]
#[case::wrong_type(r#"{"data":[{"answer":1}]}"#, false)]
#[case::missing_column(r#"{"data":[{}]}"#, false)]
#[case::exception(r#"{"data":[],"exception":"failed"}"#, false)]
#[tokio::test]
async fn typed_fetch_encodes_parameters_and_validates_rows(
    #[case] body: &str,
    #[case] valid: bool,
) {
    use litellm_storage_clickhouse::{fetch, fetch_json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string, query_param},
    };

    let server = MockServer::start().await;
    Mock::given(body_string(TypedQuery::SQL))
        .and(query_param("param_signed", i64::MIN.to_string()))
        .and(query_param("param_unsigned", u64::MAX.to_string()))
        .and(query_param("param_float", "12.5"))
        .and(query_param("param_text", "line\\nbreak"))
        .and(query_param("param_strings", "['a\\'b','雪']"))
        .and(query_param("readonly", "1"))
        .and(query_param("max_result_rows", "1000"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(2)
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).unwrap();
    let params = QueryParams {
        signed: i64::MIN,
        unsigned: u64::MAX,
        float: 12.5,
        text: "line\nbreak".into(),
        strings: vec!["a'b".into(), "雪".into()],
    };
    let rows = fetch::<TypedQuery>(&client, &connection, &params).await;
    let envelope = fetch_json::<TypedQuery>(&client, &connection, &params).await;
    if valid {
        assert_eq!(
            rows.unwrap(),
            vec![QueryRow {
                answer: "ok".into()
            }]
        );
        assert_eq!(envelope.unwrap(), body);
    } else {
        assert!(matches!(rows, Err(Error::InvalidResponse)));
        assert!(matches!(envelope, Err(Error::InvalidResponse)));
    }
}
