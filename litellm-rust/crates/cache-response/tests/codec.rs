use litellm_cache::{CacheCodec, Error};
use litellm_cache_response::{CacheEntry, ResponseCacheCodec};
use rstest::rstest;
use serde_json::{Value, json};

fn entry(response: Value) -> CacheEntry {
    CacheEntry {
        timestamp: Some(100.0),
        response,
    }
}

#[rstest]
#[case::python_literal_object(
    br#"{'timestamp': 100.0, 'response': {'text': 'hello \\ world', 'flag': True, 'empty': None, 'list': [1, 2.5]}}"#.as_slice(),
    json!({"text": "hello \\ world", "flag": true, "empty": null, "list": [1, 2.5]}),
)]
#[case::python_sync_string_response(
    br#"{'timestamp': 100.0, 'response': '{"ok": true, "text": "cached"}'}"#.as_slice(),
    json!({"ok": true, "text": "cached"}),
)]
#[case::python_literal_string_response(
    br#"{'timestamp': 100.0, 'response': "{'ok': True, 'items': (1, 2)}"}"#.as_slice(),
    json!({"ok": true, "items": [1, 2]}),
)]
#[case::json_object_response(
    br#"{"timestamp":100.0,"response":{"ok":true,"text":"cached"}}"#.as_slice(),
    json!({"ok": true, "text": "cached"}),
)]
#[case::json_string_response(
    br#"{"timestamp": 100.0, "response": "[1,2]"}"#.as_slice(),
    json!([1, 2]),
)]
fn decode_reads_every_python_envelope(#[case] bytes: &[u8], #[case] response: Value) {
    assert_eq!(ResponseCacheCodec.decode(bytes).unwrap(), entry(response));
}

#[rstest]
#[case::code_is_not_executed(b"__import__('os').system('false')".to_vec())]
#[case::non_numeric_timestamp(b"{'timestamp': 'invalid', 'response': {}}".to_vec())]
#[case::infinite_timestamp(b"{'timestamp': 1e9999, 'response': {}}".to_vec())]
#[case::missing_response(br#"{"timestamp": 100.0}"#.to_vec())]
#[case::unserialized_string_response(br#"{"timestamp": 100.0, "response": "not serialized"}"#.to_vec())]
#[case::non_utf8(vec![0xff, 0xfe])]
#[case::deep_nesting(format!("{}None{}", "[".repeat(1000), "]".repeat(1000)).into_bytes())]
fn decode_rejects_invalid_entries(#[case] bytes: Vec<u8>) {
    assert_eq!(
        ResponseCacheCodec.decode(&bytes).unwrap_err(),
        Error::InvalidEntry
    );
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinity(f64::INFINITY)]
fn encode_rejects_non_finite_timestamps(#[case] timestamp: f64) {
    assert_eq!(
        ResponseCacheCodec
            .encode(&CacheEntry {
                timestamp: Some(timestamp),
                response: json!({}),
            })
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[rstest]
#[case::object(json!({"choices": [{"text": "cached"}]}), json!({"choices": [{"text": "cached"}]}))]
#[case::array(json!([1, 2]), json!("[1,2]"))]
#[case::number(json!(7), json!("7"))]
#[case::null(json!(null), json!("null"))]
#[case::string(json!("hello world"), json!("\"hello world\""))]
#[case::numeric_string(json!("123"), json!("\"123\""))]
#[case::null_string(json!("null"), json!("\"null\""))]
fn encode_writes_python_readable_envelopes_that_round_trip(
    #[case] response: Value,
    #[case] wire_response: Value,
) {
    let wire = ResponseCacheCodec.encode(&entry(response.clone())).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&wire).unwrap(),
        json!({"timestamp": 100.0, "response": wire_response})
    );
    assert_eq!(ResponseCacheCodec.decode(&wire).unwrap(), entry(response));
}

#[rstest]
fn object_entries_preserve_the_existing_json_representation() {
    let entry = CacheEntry {
        timestamp: Some(123.0),
        response: json!({"choices": [{"text": "cached"}]}),
    };
    let bytes = ResponseCacheCodec.encode(&entry).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&entry).unwrap());
    assert_eq!(ResponseCacheCodec.decode(&bytes).unwrap(), entry);
}

#[rstest]
#[case::json(br#"{"choices": [{"text": "legacy"}]}"#.as_slice())]
#[case::python_literal(br#"{'choices': [{'text': 'legacy'}]}"#.as_slice())]
fn values_without_timestamps_decode_as_bare_responses(#[case] bytes: &[u8]) {
    assert_eq!(
        ResponseCacheCodec.decode(bytes).unwrap(),
        CacheEntry {
            timestamp: None,
            response: json!({"choices": [{"text": "legacy"}]}),
        }
    );
}
