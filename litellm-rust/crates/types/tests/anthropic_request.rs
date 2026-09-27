use litellm_types::llms::anthropic_messages::anthropic_request::{ContentBlock, ContentBlockType};
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
