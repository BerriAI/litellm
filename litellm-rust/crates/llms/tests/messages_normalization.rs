use litellm_llms::base_llm::messages::normalization::{
    fold_system_role_messages, strip_billing_metadata,
};
use litellm_llms_types::formats::messages::MessagesRequest;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text_system(json!("existing"), json!([{"type": "text", "text": "existing"}]))]
#[case::block_system(json!([{ "type": "future", "payload": 7 }]), json!([{ "type": "future", "payload": 7 }]))]
#[case::no_system(Value::Null, json!([]))]
fn folding_preserves_block_fields_order_and_unrelated_request_fields(
    #[case] system: Value,
    #[case] initial_blocks: Value,
) {
    let cache_control = json!({"type": "ephemeral", "scope": "global", "future": true});
    let folded_block = json!({"type": "text", "text": "second", "cache_control": cache_control});
    let user = json!({"role": "user", "content": "hello", "future_message": 42});
    let request: MessagesRequest = serde_json::from_value(json!({
        "model": "test-model",
        "max_tokens": 64,
        "system": system,
        "messages": [
            {"role": "system", "content": "first"},
            user,
            {"role": "system", "content": [folded_block]}
        ],
        "future_request": {"nested": true}
    }))
    .unwrap();
    let folded = fold_system_role_messages(request);
    let expected_blocks: Vec<Value> = initial_blocks
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .chain([json!({"type": "text", "text": "first"}), folded_block])
        .collect();
    assert_eq!(
        serde_json::to_value(&folded).unwrap(),
        json!({
            "model": "test-model",
            "max_tokens": 64,
            "system": expected_blocks,
            "messages": [user],
            "future_request": {"nested": true}
        })
    );
    assert_eq!(fold_system_role_messages(folded.clone()), folded);
}

const BILLING_HEADER: &str = "x-anthropic-billing-header: cc_version=1";

fn with_system(system: Value) -> MessagesRequest {
    serde_json::from_value(json!({
        "model": "test-model",
        "max_tokens": 64,
        "system": system,
        "messages": [{"role": "user", "content": "hello"}],
        "future_request": {"nested": true}
    }))
    .unwrap()
}

#[rstest]
#[case::billing_block_is_dropped_keeping_order_and_cache_control(
    json!([
        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": BILLING_HEADER},
        {"type": "text", "text": "last"}
    ]),
    json!([
        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "last"}
    ])
)]
#[case::billing_only_blocks_become_absent(json!([{"type": "text", "text": BILLING_HEADER}]), Value::Null)]
#[case::billing_string_system_becomes_absent(json!(BILLING_HEADER), Value::Null)]
#[case::plain_string_system_is_kept(json!("keep"), json!("keep"))]
#[case::billing_prefix_mid_string_is_kept(
    json!(format!("note: {BILLING_HEADER}")),
    json!(format!("note: {BILLING_HEADER}"))
)]
#[case::billing_prefix_mid_text_is_kept(
    json!([{"type": "text", "text": format!("note: {BILLING_HEADER}")}]),
    json!([{"type": "text", "text": format!("note: {BILLING_HEADER}")}])
)]
#[case::non_text_block_with_billing_text_is_kept(
    json!([{"type": "future", "text": BILLING_HEADER}]),
    json!([{"type": "future", "text": BILLING_HEADER}])
)]
fn strip_billing_metadata_drops_only_billing_text(
    #[case] system: Value,
    #[case] expected_system: Value,
) {
    let stripped = serde_json::to_value(strip_billing_metadata(with_system(system))).unwrap();
    let expected = serde_json::to_value(with_system(expected_system)).unwrap();
    assert_eq!(stripped, expected);
}
