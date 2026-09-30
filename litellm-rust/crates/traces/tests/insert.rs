use std::collections::BTreeMap;

use litellm_traces::encode_rows;
use rstest::rstest;
use serde_json::{Value, json};

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
