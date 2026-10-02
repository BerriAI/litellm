use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
};

use flate2::read::GzDecoder;
use litellm_http::Client;
use litellm_traces::Shared;
use litellm_traces_clickhouse::{
    Connection, Error, InsertRow, InsertTable, encode_rows, insert_shared_rows,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method},
};

#[fixture]
fn shared_rows(#[default(16 * 1024)] attribute_bytes: usize) -> Vec<InsertRow> {
    let resource = Shared::new(json!({"shared": "x".repeat(attribute_bytes)}));
    (0..1024)
        .map(|index| {
            BTreeMap::from([
                ("ResourceAttributes".into(), resource.clone()),
                ("SpanId".into(), Shared::new(json!(format!("{index:016x}")))),
                ("Timestamp".into(), Shared::new(json!(1))),
            ])
        })
        .collect()
}

#[rstest]
#[case::one_request(1)]
#[case::concurrent_requests(2)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_fanout_survives_gzip_insert_over_http(
    shared_rows: Vec<InsertRow>,
    #[case] concurrency: usize,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("Content-Encoding", "gzip"))
        .respond_with(ResponseTemplate::new(200))
        .expect(concurrency as u64)
        .mount(&server)
        .await;
    let client = Client::no_redirect_for_test();
    let connection = Connection::parse(&server.uri()).unwrap();
    let expected_resource = shared_rows[0]["ResourceAttributes"].clone();
    let expected_count = shared_rows.len();
    let mut requests = tokio::task::JoinSet::new();
    for _ in 0..concurrency {
        let client = client.clone();
        let connection = connection.clone();
        let rows = shared_rows.clone();
        requests.spawn(async move {
            insert_shared_rows(
                &client,
                &connection,
                "traces",
                InsertTable::OtelTraces,
                rows,
            )
            .await
        });
    }
    while let Some(result) = requests.join_next().await {
        result.unwrap().unwrap();
    }
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), concurrency);
    for request in received {
        let decoder = GzDecoder::new(request.body.as_slice());
        let mut count = 0;
        for (index, line) in BufReader::new(decoder).lines().enumerate() {
            let row: Value = serde_json::from_str(&line.unwrap()).unwrap();
            assert_eq!(&row["ResourceAttributes"], expected_resource.as_ref());
            assert_eq!(row["SpanId"], format!("{index:016x}"));
            assert_eq!(row["Timestamp"], "1970-01-01T00:00:00.000000001Z");
            assert!(row["EngineReceivedMs"].as_u64().unwrap() > 0);
            count += 1;
        }
        assert_eq!(count, expected_count);
    }
}

#[rstest]
#[tokio::test]
async fn shared_fanout_over_insert_limit_never_reaches_http(
    #[with(64 * 1024)] shared_rows: Vec<InsertRow>,
) {
    let server = MockServer::start().await;
    let connection = Connection::parse(&server.uri()).unwrap();
    let result = insert_shared_rows(
        &Client::no_redirect_for_test(),
        &connection,
        "traces",
        InsertTable::OtelTraces,
        shared_rows,
    )
    .await;
    assert!(matches!(result, Err(Error::InsertTooLarge)));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::span("Timestamp", json!(1_234_567_890), json!("1970-01-01T00:00:01.23456789Z"))]
#[case::start("start_time", json!(1_234), json!("1970-01-01T00:00:01.234Z"))]
#[case::end("end_time", json!(2_345), json!("1970-01-01T00:00:02.345Z"))]
#[case::completion("completion_start_time", json!(1_345), json!("1970-01-01T00:00:01.345Z"))]
#[case::absent_completion("completion_start_time", Value::Null, Value::Null)]
#[case::before_epoch("Timestamp", json!(-1), json!("1969-12-31T23:59:59.999999999Z"))]
fn insert_encoding_preserves_timestamp_precision_and_other_fields(
    #[case] field: &str,
    #[case] value: Value,
    #[case] expected: Value,
) {
    let rows = vec![BTreeMap::from([
        (field.to_owned(), value),
        ("SpanAttributes".into(), json!({"message": "a\nb\\c\"雪"})),
        ("InputTokens".into(), json!(42)),
    ])];
    let encoded = encode_rows(rows).expect("valid row");
    let actual: Value = serde_json::from_str(&encoded).expect("JSONEachRow record");
    assert_eq!(
        actual,
        json!({
            field: expected, "SpanAttributes": {"message": "a\nb\\c\"雪"}, "InputTokens": 42
        })
    );
}

#[rstest]
#[case::fractional(json!(1.25))]
#[case::out_of_range(json!(u64::MAX))]
#[case::null(Value::Null)]
fn insert_encoding_rejects_invalid_span_timestamps(#[case] timestamp: Value) {
    assert!(encode_rows(vec![BTreeMap::from([("Timestamp".into(), timestamp)])]).is_err());
}
