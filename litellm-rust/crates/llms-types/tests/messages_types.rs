use litellm_llms_types::formats::messages::{
    AppliedEdit, BlockContent, BuiltinMessagesTool, Citations, ContainerReference,
    ContentBlockPayload, ContentSource, ContextManagementResponse, ContextTrigger, CustomTool,
    McpServer, MessageRole, MessageType, MessagesCompaction, MessagesContainer, MessagesMetadata,
    MessagesUsage, OutputFormat, PromptCacheBreakpoint, Safeguard, StopDetails, StopReason,
    ToolCaller, ToolChoice, ToolDefinition, WebSearchResultError,
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

#[rstest]
fn custom_tool_exposes_schema_and_preserves_extensions() {
    let wire = json!({
        "name":"lookup",
        "input_schema":{"type":"object","properties":{"query":{"type":"string"}}},
        "strict":false,
        "extension":{"nested":[1,null]}
    });
    let tool: CustomTool = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(tool.definition.name.as_deref(), Some("lookup"));
    assert_eq!(tool.definition.strict, Some(false));
    let Some(litellm_llms_types::json_schema::JsonSchema::Object(schema)) =
        &tool.definition.input_schema
    else {
        panic!("expected an object schema");
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("query"));
    assert_eq!(serde_json::to_value(tool).unwrap(), wire);
}

#[rstest]
#[case::missing_name(json!({"input_schema":{}}))]
#[case::null_name(json!({"name":null}))]
#[case::wrong_name_shape(json!({"name":7}))]
#[case::tagged_tool(json!({"name":"lookup","type":"custom"}))]
#[case::null_discriminator(json!({"name":"lookup","type":null}))]
fn custom_tool_rejects_invalid_shapes(#[case] wire: Value) {
    assert!(serde_json::from_value::<CustomTool>(wire).is_err());
}

#[rstest]
#[case::advisor("advisor_20260301", BuiltinMessagesTool::Advisor)]
#[case::toolsearchregex(
    "tool_search_tool_regex_20251119",
    BuiltinMessagesTool::ToolSearchRegex
)]
#[case::toolsearchbm25("tool_search_tool_bm25_20251119", BuiltinMessagesTool::ToolSearchBm25)]
#[case::custom("custom", BuiltinMessagesTool::Custom)]
#[case::websearch("web_search_20250305", BuiltinMessagesTool::WebSearch)]
#[case::computer("computer_20250124", BuiltinMessagesTool::Computer)]
#[case::bash("bash_20250124", BuiltinMessagesTool::Bash)]
#[case::texteditor("text_editor_20250728", BuiltinMessagesTool::TextEditor)]
#[case::codeexecution("code_execution_20250825", BuiltinMessagesTool::CodeExecution)]
#[case::websearch20260209("web_search_20260209", BuiltinMessagesTool::WebSearch20260209)]
#[case::computer20241022("computer_20241022", BuiltinMessagesTool::Computer20241022)]
#[case::bash20241022("bash_20241022", BuiltinMessagesTool::Bash20241022)]
#[case::texteditor20241022("text_editor_20241022", BuiltinMessagesTool::TextEditor20241022)]
#[case::texteditor20250124("text_editor_20250124", BuiltinMessagesTool::TextEditor20250124)]
#[case::codeexecution20250522(
    "code_execution_20250522",
    BuiltinMessagesTool::CodeExecution20250522
)]
#[case::memory("memory_20250818", BuiltinMessagesTool::Memory)]
#[case::webfetch("web_fetch_20250910", BuiltinMessagesTool::WebFetch)]
#[case::webfetch20260209("web_fetch_20260209", BuiltinMessagesTool::WebFetch20260209)]
#[case::webfetch20260309("web_fetch_20260309", BuiltinMessagesTool::WebFetch20260309)]
#[case::webfetch20260318("web_fetch_20260318", BuiltinMessagesTool::WebFetch20260318)]
#[case::websearch20260318("web_search_20260318", BuiltinMessagesTool::WebSearch20260318)]
#[case::codeexecution20260120(
    "code_execution_20260120",
    BuiltinMessagesTool::CodeExecution20260120
)]
#[case::codeexecution20260521(
    "code_execution_20260521",
    BuiltinMessagesTool::CodeExecution20260521
)]
#[case::computer20251124("computer_20251124", BuiltinMessagesTool::Computer20251124)]
#[case::texteditor20250429("text_editor_20250429", BuiltinMessagesTool::TextEditor20250429)]
fn builtin_tools_decode_typed_definitions(
    #[case] tag: &str,
    #[case] constructor: fn(ToolDefinition) -> BuiltinMessagesTool,
) {
    let wire = json!({"type":tag,"name":"lookup","max_uses":3,"extension":[1,null]});
    let tool: BuiltinMessagesTool = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        tool,
        constructor(ToolDefinition {
            name: Some("lookup".into()),
            max_uses: Some(3),
            extra: serde_json::Map::from_iter([("extension".into(), json!([1, null]))]),
            ..Default::default()
        })
    );
    assert_eq!(serde_json::to_value(tool).unwrap(), wire);
}

#[rstest]
fn compaction_trigger_exposes_token_threshold() {
    let wire = json!({"type":"input_tokens","value":1024,"extension":true});
    let trigger: ContextTrigger = serde_json::from_value(wire.clone()).unwrap();
    let ContextTrigger::InputTokens { value, .. } = &trigger;
    assert_eq!(*value, 1024);
    assert_eq!(serde_json::to_value(trigger).unwrap(), wire);
}

#[rstest]
#[case::negative(json!({"type":"input_tokens","value":-1}))]
#[case::wrong_shape(json!({"type":"input_tokens","value":"1024"}))]
#[case::missing_value(json!({"type":"input_tokens"}))]
fn token_threshold_requires_unsigned_integer(#[case] wire: Value) {
    assert!(serde_json::from_value::<ContextTrigger>(wire).is_err());
}

#[rstest]
#[case::tool_result(json!({"content":[{"type":"text","text":"found"}],"is_error":false,"caller":{"type":"direct"}}))]
#[case::web_search(json!({"url":"https://example.test","title":"result","page_age":"today","encrypted_content":"opaque","citations":[{"type":"web_search_result_location","url":"https://example.test"}]}))]
#[case::code_execution(json!({"stdout":"done","stderr":"","return_code":0,"encrypted_stdout":"opaque"}))]
#[case::text_editor(json!({"file_type":"text","num_lines":2,"start_line":1,"total_lines":2,"is_file_update":false,"lines":["a","b"],"new_lines":1,"new_start":2,"old_lines":1,"old_start":2}))]
#[case::tool_reference(json!({"tool":{"type":"tool_use","name":"lookup","input":{"query":[1,null]}},"tool_references":[{"type":"tool_reference","tool_name":"lookup"}]}))]
#[case::document(json!({"source":{"type":"text","media_type":"text/plain","data":"document"},"context":"context","prompt_cache_breakpoint":{"mode":"explicit"},"extension":{"nested":[1,null]}}))]
fn content_payload_families_round_trip(#[case] wire: Value) {
    round_trip::<ContentBlockPayload>(wire);
}

#[rstest]
fn content_payload_exposes_sources_results_and_execution_fields() {
    let wire = json!({
        "source":{"type":"url","url":"https://example.test/document"},
        "content":{"type":"web_search_tool_result_error","error_code":"unavailable"},
        "caller":{"type":"code_execution_20250825","tool_id":"call_1"},
        "return_code":-1,
        "is_error":true
    });
    let payload: ContentBlockPayload = serde_json::from_value(wire.clone()).unwrap();
    let Some(ContentSource::Url { url, .. }) = &payload.source else {
        panic!("expected URL source");
    };
    assert_eq!(url, "https://example.test/document");
    let Some(BlockContent::SearchError(error)) = &payload.content else {
        panic!("expected search error");
    };
    assert_eq!(error.error_code.as_deref(), Some("unavailable"));
    assert!(matches!(
        payload.caller,
        Some(ToolCaller::CodeExecution { .. })
    ));
    assert_eq!(payload.return_code, Some(-1));
    assert_eq!(payload.is_error, Some(true));
    assert_eq!(serde_json::to_value(payload).unwrap(), wire);
}
