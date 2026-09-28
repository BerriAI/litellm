use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessagesRequest, ContentBlock, ContentBlockType,
};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::text("text", ContentBlockType::Text)]
#[case::thinking("thinking", ContentBlockType::Thinking)]
#[case::redacted_thinking("redacted_thinking", ContentBlockType::RedactedThinking)]
#[case::tool_use("tool_use", ContentBlockType::ToolUse)]
#[case::server_tool_use("server_tool_use", ContentBlockType::ServerToolUse)]
#[case::tool_result("tool_result", ContentBlockType::ToolResult)]
#[case::compaction("compaction", ContentBlockType::Compaction)]
#[case::advisor_result("advisor_tool_result", ContentBlockType::AdvisorToolResult)]
#[case::web_search_result("web_search_tool_result", ContentBlockType::WebSearchToolResult)]
#[case::future_block("future_block", ContentBlockType::Other("future_block".into()))]
#[case::case_sensitive("Tool_Use", ContentBlockType::Other("Tool_Use".into()))]
#[case::empty("", ContentBlockType::Other(String::new()))]
fn block_type_is_typed_and_round_trips_with_extra_fields(
    #[case] wire: &str,
    #[case] expected: ContentBlockType,
) {
    let input = json!({"type": wire, "future_field": {"nested": [1, null]}});
    let block: ContentBlock = serde_json::from_value(input.clone()).unwrap();
    assert_eq!(block.block_type.as_ref(), Some(&expected));
    assert_eq!(serde_json::to_value(block).unwrap(), input);
}

#[rstest]
#[case::number(json!(1))]
#[case::boolean(json!(true))]
#[case::array(json!(["text"]))]
#[case::enum_object(json!({"text": null}))]
fn block_type_rejects_non_string_values(#[case] value: Value) {
    assert!(serde_json::from_value::<ContentBlock>(json!({"type": value})).is_err());
}

#[rstest]
#[case::same_type(json!({"type": "tool_use"}), ContentBlockType::ToolUse, true)]
#[case::other_type(json!({"type": "tool_result"}), ContentBlockType::ToolUse, false)]
#[case::unknown_type(json!({"type": "future_tool"}), ContentBlockType::ToolUse, false)]
#[case::no_type(json!({"text": "x"}), ContentBlockType::Text, false)]
fn is_type_matches_the_exact_block_type(
    #[case] block: Value,
    #[case] block_type: ContentBlockType,
    #[case] expected: bool,
) {
    let block: ContentBlock = serde_json::from_value(block).unwrap();
    assert_eq!(block.is_type(block_type), expected);
}

#[rstest]
#[case::absent(json!({}), json!({"temperature": 0.5, "metadata": {"old": true}, "future": {"old": true}}))]
#[case::replace(json!({"temperature": 0, "metadata": {"new": true}, "future": {"new": null}}), json!({"temperature": 0.0, "metadata": {"new": true}, "future": {"new": null}}))]
#[case::clear(json!({"temperature": null, "future": null}), json!({"metadata": {"old": true}, "future": null}))]
fn request_overrides_preserve_absent_fields_and_replace_supplied_fields(
    #[case] overrides: Value,
    #[case] expected: Value,
) {
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "test-model", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5, "metadata": {"old": true}, "future": {"old": true}
    }))
    .unwrap();
    let updated = request
        .clone()
        .with_overrides(overrides.as_object().unwrap().clone())
        .unwrap();
    assert_eq!(updated.model, request.model);
    assert_eq!(updated.messages, request.messages);
    assert_eq!(serde_json::to_value(updated.params).unwrap(), expected);
}

#[rstest]
#[case::invalid_optional(json!({"temperature": "hot"}))]
#[case::invalid_required(json!({"messages": null}))]
fn request_overrides_validate_supplied_fields(#[case] overrides: Value) {
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "test-model", "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(
        request
            .with_overrides(overrides.as_object().unwrap().clone())
            .is_err()
    );
}
