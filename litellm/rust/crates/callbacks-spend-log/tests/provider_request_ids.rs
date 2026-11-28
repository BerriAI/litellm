//! The `provider_request_ids` metadata stamp. The recorded goldens cannot hold it: the frozen
//! Python builder they replay does not stamp it, so every golden row carries `null`.

use std::sync::Arc;

use litellm_callbacks_spend_log::{RowConfig, RowInput, SpendLogRowBuilder};
use litellm_redaction::TokenRedactor;
use rstest::rstest;
use serde_json::{Value, json};

const TOKEN_SPLIT: &str = "#~#";

fn golden(name: &str) -> String {
    std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/goldens")
            .join(name),
    )
    .expect("golden reads")
    .replace(TOKEN_SPLIT, "")
}

/// The input of the first recorded row, a plain completion with no response headers.
fn template() -> Value {
    let line = golden("rows.jsonl")
        .lines()
        .next()
        .expect("a recorded row")
        .to_owned();
    let mut record: Value = serde_json::from_str(&line).expect("golden parses");
    record["input"].take()
}

fn stored_ids(mut input: Value, response_headers: Value, additional_headers: Value) -> Value {
    input["kwargs"]["response_headers"] = response_headers;
    input["standard_logging_object"]["hidden_params"] =
        json!({ "additional_headers": additional_headers });
    stored_metadata(&input)["provider_request_ids"].clone()
}

fn stored_metadata(input: &Value) -> Value {
    let ranges: Vec<(u32, u32)> =
        serde_json::from_str(&golden("token_word_ranges.json")).expect("word ranges parse");
    let config = RowConfig {
        latest_turn_only: false,
        redact_tokens: false,
        drop_requester_ip_address: false,
        omit_missing_session_id: false,
        max_string_length_prompt_in_db: 2048,
        disable_end_user_cost_tracking: false,
        request_correlation_in_logs: false,
        turn_off_message_logging: false,
    };
    let builder = SpendLogRowBuilder::new(config, Arc::new(TokenRedactor::new(16, ranges)));
    let row = builder
        .build_row(&RowInput::from_json(&input.to_string()).expect("input parses"))
        .expect("row builds");
    let columns = serde_json::to_value(&row).expect("row serializes");
    serde_json::from_str(columns["metadata"].as_str().expect("metadata is text"))
        .expect("metadata parses")
}

#[test]
fn a_response_without_request_ids_stores_null() {
    assert_eq!(
        stored_ids(template(), Value::Null, Value::Null),
        Value::Null
    );
}

#[test]
fn raw_response_headers_are_stored_by_provider_header_name() {
    let stored = stored_ids(
        template(),
        json!({ "apim-request-id": "apim-1", "x-request-id": "req-1", "content-type": "x" }),
        Value::Null,
    );
    assert_eq!(
        stored,
        json!({ "apim-request-id": "apim-1", "x-request-id": "req-1" })
    );
}

#[test]
fn streaming_calls_keep_their_ids_in_the_llm_provider_copies() {
    let stored = stored_ids(
        template(),
        Value::Null,
        json!({ "llm_provider-apim-request-id": "apim-2" }),
    );
    assert_eq!(stored, json!({ "apim-request-id": "apim-2" }));
}

#[test]
fn the_llm_provider_copy_wins_when_both_carry_the_same_header() {
    let stored = stored_ids(
        template(),
        json!({ "apim-request-id": "raw" }),
        json!({ "llm_provider-apim-request-id": "copy" }),
    );
    assert_eq!(stored, json!({ "apim-request-id": "copy" }));
}

#[test]
fn header_names_match_without_regard_to_case() {
    let stored = stored_ids(
        template(),
        json!({ "APIM-Request-Id": "apim-3" }),
        Value::Null,
    );
    assert_eq!(stored, json!({ "apim-request-id": "apim-3" }));
}

#[rstest]
#[case::empty(String::new())]
#[case::too_long("x".repeat(257))]
#[case::not_ascii("réq-1".to_owned())]
fn an_unusable_id_is_not_stored(#[case] id: String) {
    let stored = stored_ids(template(), json!({ "apim-request-id": id }), Value::Null);
    assert_eq!(stored, Value::Null);
}

#[test]
fn an_id_at_the_length_limit_is_stored() {
    let id = "x".repeat(256);
    let stored = stored_ids(template(), json!({ "apim-request-id": id }), Value::Null);
    assert_eq!(stored, json!({ "apim-request-id": id }));
}

#[test]
fn a_non_string_value_is_not_stored() {
    let stored = stored_ids(template(), json!({ "apim-request-id": 7 }), Value::Null);
    assert_eq!(stored, Value::Null);
}

#[test]
fn a_caller_cannot_forge_the_stamp_through_request_metadata() {
    let mut input = template();
    input["metadata"]["provider_request_ids"] = json!({ "apim-request-id": "forged" });
    input["litellm_params"]["metadata"]["provider_request_ids"] =
        json!({ "apim-request-id": "forged" });
    assert_eq!(stored_metadata(&input)["provider_request_ids"], Value::Null);
}
