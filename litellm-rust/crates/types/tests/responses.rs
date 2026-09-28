use litellm_types::responses::main::ResponsesApiRequest;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text(json!({"model":"test-model", "input":"hello", "future_option":null}))]
#[case::items(json!({"model":"test-model", "input":[{"type":"future_item", "payload":42}], "stream":true}))]
#[case::non_streaming(json!({"model":"test-model", "input":[], "stream":false}))]
fn request_round_trip_preserves_fields(#[case] body: Value) {
    let request: ResponsesApiRequest = serde_json::from_value(body.clone()).unwrap();
    assert_eq!(serde_json::to_value(request).unwrap(), body);
}

#[rstest]
#[case::model(json!({"model":42, "input":"hello"}))]
#[case::input(json!({"model":"test-model", "input":{}}))]
#[case::missing_input(json!({"model":"test-model"}))]
#[case::stream(json!({"model":"test-model", "input":"hello", "stream":"true"}))]
#[case::null_stream(json!({"model":"test-model", "input":"hello", "stream":null}))]
fn request_rejects_malformed_known_fields(#[case] body: Value) {
    assert!(serde_json::from_value::<ResponsesApiRequest>(body).is_err());
}
