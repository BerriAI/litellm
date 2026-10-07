use litellm_llms_types::formats::messages::{
    AppliedEdit, BlockContent, Citations, ContainerReference, ContentSource,
    ContextManagementResponse, McpServer, MessageRole, MessageType, MessagesCompaction,
    MessagesContainer, MessagesMetadata, MessagesUsage, OutputFormat, PromptCacheBreakpoint,
    Safeguard, StopDetails, StopReason, ToolCaller, ToolChoice, ToolDefinition,
    WebSearchResultError,
};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value)
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
fn metadata_contracts_round_trip() {
    round_trip::<MessagesMetadata>(json!({"user_id":"user_1","future":true}));
    round_trip::<OutputFormat>(json!({
        "type":"json_schema",
        "schema":{"type":"object","properties":{"name":{"type":"string"}}},
        "strict":true
    }));
    round_trip::<MessagesCompaction>(json!({"type":"summarize","instructions":"briefly"}));
    round_trip::<MessagesContainer>(json!({
        "id":"container_1",
        "expires_at":"2026-01-01T00:00:00Z",
        "skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]
    }));
    round_trip::<ContainerReference>(json!({"id":"container_1"}));
    round_trip::<ContainerReference>(json!({"id":"container_1","skills":[{"type":"anthropic"}]}));
    round_trip::<McpServer>(json!({
        "type":"url",
        "url":"https://example.test/mcp",
        "name":"search",
        "authorization_token":"token",
        "tool_configuration":{"allowed_tools":["search"],"enabled":true}
    }));
    round_trip::<StopDetails>(
        json!({"type":"refusal","category":"safety","explanation":"blocked"}),
    );
    round_trip::<ContextManagementResponse>(json!({
        "applied_edits":[{"type":"compact_20260112","summary_input_tokens":7,"warnings":["notice"]}]
    }));
    round_trip::<AppliedEdit>(json!({"type":"clear","cleared_input_tokens":3}));
    round_trip::<Safeguard>(json!({"type":"classifier","classifier_context":{"source":"test"}}));
}

#[rstest]
#[case::user(json!("user"))]
#[case::assistant(json!("assistant"))]
#[case::system(json!("system"))]
#[case::future(json!("future_role"))]
fn message_roles_round_trip(#[case] wire: Value) {
    round_trip::<MessageRole>(wire);
}

#[rstest]
#[case::message(json!("message"))]
#[case::future(json!("future_type"))]
fn message_types_round_trip(#[case] wire: Value) {
    round_trip::<MessageType>(wire);
}

#[rstest]
#[case::end_turn(json!("end_turn"))]
#[case::refusal(json!("refusal"))]
#[case::compaction(json!("compaction"))]
#[case::future(json!("future_reason"))]
fn stop_reasons_round_trip(#[case] wire: Value) {
    round_trip::<StopReason>(wire);
}

#[rstest]
fn tool_contracts_round_trip() {
    round_trip::<ToolDefinition>(json!({
        "name":"search",
        "description":"Search the web",
        "input_schema":{"type":"object","properties":{"query":{"type":"string"}}},
        "citations":{"enabled":true},
        "user_location":{"type":"approximate","city":"San Francisco","country":"US"}
    }));
    round_trip::<ToolChoice>(
        json!({"type":"tool","name":"search","disable_parallel_tool_use":true}),
    );
}

#[rstest]
fn usage_contracts_round_trip() {
    round_trip::<MessagesUsage>(json!({
        "input_tokens":10,
        "output_tokens":4,
        "server_tool_use":{"web_search_requests":2,"web_fetch_requests":1},
        "cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":1},
        "output_tokens_details":{"thinking_tokens":2},
        "iterations":[
            {"type":"compaction","input_tokens":7,"output_tokens":1},
            {"type":"message","input_tokens":3,"output_tokens":3}
        ],
        "service_tier":"priority",
        "speed":"fast"
    }));
}

#[rstest]
#[case::base64(json!({"type":"base64","media_type":"image/png","data":"AA=="}))]
#[case::url(json!({"type":"url","url":"https://example.test/image"}))]
#[case::file(json!({"type":"file","file_id":"file_1"}))]
#[case::text(json!({"type":"text","media_type":"text/plain","data":"document"}))]
#[case::content(json!({"type":"content","content":"nested"}))]
fn content_sources_round_trip(#[case] wire: Value) {
    round_trip::<ContentSource>(wire);
}

#[rstest]
fn content_block_and_tool_caller_round_trip() {
    round_trip::<BlockContent>(json!("text"));
    round_trip::<ToolCaller>(json!({"type":"code_execution_20250825","tool_id":"server_1"}));
    round_trip::<ToolCaller>(json!({"type":"direct"}));
}

#[rstest]
#[case::configuration(json!({"enabled":true,"future":null}))]
#[case::page_citation(json!([{"type":"page_location","cited_text":"quote","document_index":0,"start_page_number":1}] ))]
#[case::character_citation(json!([{"type":"char_location","cited_text":"quote","document_index":0,"start_char_index":1,"end_char_index":6}] ))]
#[case::web_search_citation(json!([{"type":"web_search_result_location","url":"https://example.test","title":"result"}]))]
#[case::content_block_citation(json!([{"type":"content_block_location","document_index":0,"start_block_index":1,"end_block_index":2}]))]
#[case::search_result_citation(json!([{"type":"search_result_location","search_result_index":0,"source":"web"}]))]
fn citation_forms_round_trip(#[case] wire: Value) {
    round_trip::<Citations>(wire);
}

#[rstest]
fn prompt_cache_and_search_error_round_trip() {
    round_trip::<PromptCacheBreakpoint>(json!({"mode":"explicit"}));
    round_trip::<WebSearchResultError>(
        json!({"type":"web_search_tool_result_error","error_code":"unavailable"}),
    );
}
