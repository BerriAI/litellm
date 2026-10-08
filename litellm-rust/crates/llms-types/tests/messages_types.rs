use litellm_llms_types::formats::messages::{
    AdvisorToolResultContent, AppliedEdit, BlockContent, BrowserStateChange, BuiltinMessagesTool,
    Citation, CodeExecutionOutput, CodeExecutionToolResultContent, ContainerReference,
    ContentSource, ContextManagementResponse, ContextTrigger, CustomTool, CustomToolType,
    McpServer, MessageRole, MessageType, MessagesCompaction, MessagesContainer,
    MessagesContentPart, MessagesMetadata, MessagesToolParam, MessagesUsage, OutputFormat,
    PromptCacheBreakpoint, Safeguard, StopDetails, StopReason,
    TextEditorCodeExecutionToolResultContent, TextEditorFileType, ToolCaller, ToolChange,
    ToolChangeTarget, ToolChoice, ToolDefinition, ToolSearchToolResultContent,
    WebFetchToolResultContent, WebSearchToolResultContent,
};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

fn round_trip<T>(wire: Value) -> T
where
    T: DeserializeOwned + Serialize,
{
    let parsed: T = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
    parsed
}

#[rstest]
fn metadata_contracts_round_trip() {
    let metadata = round_trip::<MessagesMetadata>(json!({"user_id":"user_1","future":true}));
    assert_eq!(metadata.user_id.as_deref(), Some("user_1"));
    assert_eq!(metadata.extra.get("future"), Some(&json!(true)));
    let output_format = round_trip::<OutputFormat>(json!({
        "type":"json_schema",
        "schema":{"type":"object","properties":{"name":{"type":"string"}}},
        "strict":true
    }));
    assert_eq!(output_format.strict, Some(true));
    let Some(litellm_llms_types::json_schema::JsonSchema::Object(schema)) = &output_format.schema
    else {
        panic!("expected output object schema");
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("name"));
    let compaction =
        round_trip::<MessagesCompaction>(json!({"type":"summarize","instructions":"briefly"}));
    assert_eq!(compaction.instructions.as_deref(), Some("briefly"));
    let container = round_trip::<MessagesContainer>(json!({
        "id":"container_1",
        "expires_at":"2026-01-01T00:00:00Z",
        "skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]
    }));
    assert_eq!(container.id.as_deref(), Some("container_1"));
    let [skill] = container.skills.as_ref().unwrap().as_slice() else {
        panic!("expected container skill");
    };
    assert_eq!(skill.skill_id.as_deref(), Some("skill_1"));
    assert_eq!(skill.version.as_deref(), Some("1"));
    let reference = round_trip::<ContainerReference>(json!({"id":"container_1"}));
    let ContainerReference::Parameters(parameters) = reference else {
        panic!("expected container parameters");
    };
    assert_eq!(parameters.id.as_deref(), container.id.as_deref());
    assert!(parameters.skills.is_none());
    round_trip::<ContainerReference>(json!({"id":"container_1","skills":[{"type":"anthropic"}]}));
    let server = round_trip::<McpServer>(json!({
        "type":"url",
        "url":"https://example.test/mcp",
        "name":"search",
        "authorization_token":"token",
        "tool_configuration":{"allowed_tools":["search"],"enabled":true}
    }));
    assert_eq!(server.name.as_deref(), Some("search"));
    let configuration = server.tool_configuration.as_ref().unwrap();
    assert_eq!(configuration.enabled, Some(true));
    assert_eq!(
        configuration.allowed_tools.as_deref(),
        Some([String::from("search")].as_slice())
    );
    let stop = round_trip::<StopDetails>(
        json!({"type":"refusal","category":"safety","explanation":"blocked"}),
    );
    assert_eq!(stop.category.as_deref(), Some("safety"));
    assert_eq!(stop.explanation.as_deref(), Some("blocked"));
    let context = round_trip::<ContextManagementResponse>(json!({
        "applied_edits":[{"type":"compact_20260112","summary_input_tokens":7,"warnings":["notice"]}]
    }));
    let [edit] = context.applied_edits.as_ref().unwrap().as_slice() else {
        panic!("expected applied edit");
    };
    assert_eq!(edit.summary_input_tokens, Some(7));
    assert_eq!(
        edit.warnings.as_deref(),
        Some([String::from("notice")].as_slice())
    );
    let cleared = round_trip::<AppliedEdit>(json!({"type":"clear","cleared_input_tokens":3}));
    assert_eq!(cleared.cleared_input_tokens, Some(3));
    let safeguard = round_trip::<Safeguard>(
        json!({"type":"classifier","classifier_context":{"source":"test"}}),
    );
    assert_eq!(safeguard.safeguard_type, "classifier");
    assert_eq!(
        safeguard.classifier_context.as_ref().unwrap().get("source"),
        Some(&json!("test"))
    );
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
    let definition = round_trip::<ToolDefinition>(json!({
        "name":"search",
        "description":"Search the web",
        "input_schema":{"type":"object","properties":{"query":{"type":"string"}}},
        "citations":{"enabled":true},
        "user_location":{"type":"approximate","city":"San Francisco","country":"US"}
    }));
    assert_eq!(definition.name.as_deref(), Some("search"));
    assert_eq!(definition.citations.as_ref().unwrap().enabled, Some(true));
    assert_eq!(
        definition.user_location.as_ref().unwrap().city.as_deref(),
        Some("San Francisco")
    );
    let choice = round_trip::<ToolChoice>(
        json!({"type":"tool","name":"search","disable_parallel_tool_use":true}),
    );
    assert_eq!(choice.name, definition.name);
    assert_eq!(choice.disable_parallel_tool_use, Some(true));
}

#[rstest]
fn usage_contracts_round_trip() {
    let usage = round_trip::<MessagesUsage>(json!({
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
        "inference_geo":"global",
        "speed":"fast"
    }));
    assert_eq!(usage.inference_geo.as_deref(), Some("global"));
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(4));
    let server = usage.server_tool_use.as_ref().unwrap();
    assert_eq!(server.web_search_requests, Some(2));
    assert_eq!(server.web_fetch_requests, Some(1));
    let cache = usage.cache_creation.as_ref().unwrap();
    assert_eq!(cache.ephemeral_1h_input_tokens, Some(3));
    assert_eq!(cache.ephemeral_5m_input_tokens, Some(1));
    assert_eq!(
        usage
            .output_tokens_details
            .as_ref()
            .unwrap()
            .thinking_tokens,
        Some(2)
    );
    let [compaction, message] = usage.iterations.as_ref().unwrap().as_slice() else {
        panic!("expected usage iterations");
    };
    assert_eq!(
        compaction.iteration_type,
        litellm_llms_types::formats::messages::UsageIterationType::Compaction
    );
    assert_eq!(
        message.iteration_type,
        litellm_llms_types::formats::messages::UsageIterationType::Message
    );
    assert_eq!(
        compaction.input_tokens.unwrap() + message.input_tokens.unwrap(),
        usage.input_tokens.unwrap()
    );
    assert_eq!(
        compaction.output_tokens.unwrap() + message.output_tokens.unwrap(),
        usage.output_tokens.unwrap()
    );
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
#[case::advisor("advisor_20260301", BuiltinMessagesTool::Advisor)]
#[case::toolsearchregex(
    "tool_search_tool_regex_20251119",
    BuiltinMessagesTool::ToolSearchRegex
)]
#[case::toolsearchbm25("tool_search_tool_bm25_20251119", BuiltinMessagesTool::ToolSearchBm25)]
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
#[case::toolsearchregexlatest("tool_search_tool_regex", BuiltinMessagesTool::ToolSearchRegexLatest)]
#[case::toolsearchbm25latest("tool_search_tool_bm25", BuiltinMessagesTool::ToolSearchBm25Latest)]
#[case::browsertoolset("browser_toolset_20260801", BuiltinMessagesTool::BrowserToolset)]
#[case::computertoolset("computer_toolset_20260801", BuiltinMessagesTool::ComputerToolset)]
#[case::mcptoolset("mcp_toolset", BuiltinMessagesTool::McpToolset)]
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
#[case::input_tokens("input_tokens", false)]
#[case::tool_uses("tool_uses", true)]
fn context_trigger_exposes_typed_threshold(#[case] tag: &str, #[case] tool_uses: bool) {
    let wire = json!({"type":tag,"value":1024,"extension":true});
    let trigger: ContextTrigger = serde_json::from_value(wire.clone()).unwrap();
    match &trigger {
        ContextTrigger::InputTokens { value, extra } => {
            assert!(!tool_uses);
            assert_eq!(*value, 1024);
            assert_eq!(extra.get("extension"), Some(&json!(true)));
        }
        ContextTrigger::ToolUses { value, extra } => {
            assert!(tool_uses);
            assert_eq!(*value, 1024);
            assert_eq!(extra.get("extension"), Some(&json!(true)));
        }
    }
    assert_eq!(serde_json::to_value(trigger).unwrap(), wire);
}

#[rstest]
#[case::negative(json!({"type":"input_tokens","value":-1}))]
#[case::wrong_shape(json!({"type":"input_tokens","value":"1024"}))]
#[case::missing_value(json!({"type":"input_tokens"}))]
#[case::null_value(json!({"type":"input_tokens","value":null}))]
#[case::tool_uses_negative(json!({"type":"tool_uses","value":-1}))]
#[case::tool_uses_fractional(json!({"type":"tool_uses","value":1.5}))]
#[case::tool_uses_missing_value(json!({"type":"tool_uses"}))]
#[case::missing_discriminator(json!({"value":1}))]
#[case::unknown_discriminator(json!({"type":"other","value":1}))]
fn token_threshold_requires_unsigned_integer(#[case] wire: Value) {
    assert!(serde_json::from_value::<ContextTrigger>(wire).is_err());
}

#[rstest]
#[case::missing_discriminator(json!({"name":"lookup"}))]
#[case::unknown_discriminator(json!({"type":"future_tool"}))]
#[case::null_discriminator(json!({"type":null}))]
#[case::wrong_description(json!({"type":"bash_20250124","description":7}))]
#[case::wrong_schema(json!({"type":"bash_20250124","input_schema":[]}))]
#[case::negative_limit(json!({"type":"web_search_20250305","max_uses":-1}))]
fn builtin_tools_reject_malformed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<BuiltinMessagesTool>(wire).is_err());
}

#[rstest]
fn builtin_tools_accept_partial_definitions() {
    let wire = json!({"type":"bash_20250124","extension":null});
    let tool: BuiltinMessagesTool = serde_json::from_value(wire.clone()).unwrap();
    let BuiltinMessagesTool::Bash(definition) = &tool else {
        panic!("expected bash tool");
    };
    assert!(definition.name.is_none());
    assert!(definition.input_schema.is_none());
    assert!(definition.max_uses.is_none());
    assert_eq!(definition.extra.get("extension"), Some(&Value::Null));
    assert_eq!(serde_json::to_value(tool).unwrap(), wire);
}

#[rstest]
fn existing_content_blocks_preserve_opaque_nested_fields() {
    let wire =
        json!({"type":"tool_result","content":[{"type":"text","text":7}],"source":{"url":7}});
    let block: litellm_llms_types::formats::messages::ContentBlock =
        serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(block.content.as_ref(), wire.get("content"));
    assert_eq!(block.extra.get("source"), wire.get("source"));
    assert_eq!(serde_json::to_value(block).unwrap(), wire);
}

fn part(wire: Value) -> MessagesContentPart {
    round_trip::<MessagesContentPart>(wire)
}

#[rstest]
fn text_blocks_expose_every_citation_location() {
    let block = part(json!({
        "type":"text",
        "text":"cited",
        "citations":[
            {"type":"char_location","cited_text":"a","document_index":0,"start_char_index":1,"end_char_index":6,"file_id":"file_1"},
            {"type":"page_location","cited_text":"b","document_index":1,"start_page_number":1,"end_page_number":2},
            {"type":"content_block_location","cited_text":"c","document_index":2,"start_block_index":0,"end_block_index":1},
            {"type":"web_search_result_location","cited_text":"d","url":"https://example.test","encrypted_index":"opaque","title":"result"},
            {"type":"search_result_location","cited_text":"e","search_result_index":3,"source":"kb","start_block_index":0,"end_block_index":2}
        ],
        "cache_control":{"type":"ephemeral"}
    }));
    let MessagesContentPart::Text(text) = &block else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "cited");
    let [
        Citation::CharLocation(chars),
        Citation::PageLocation(page),
        Citation::ContentBlockLocation(blocks),
        Citation::WebSearchResultLocation(search),
        Citation::SearchResultLocation(result),
    ] = text.citations.as_deref().unwrap()
    else {
        panic!("expected one citation of each location type");
    };
    assert_eq!((chars.start_char_index, chars.end_char_index), (1, 6));
    assert_eq!(chars.file_id.as_deref(), Some("file_1"));
    assert_eq!((page.start_page_number, page.end_page_number), (1, 2));
    assert_eq!(blocks.document_index, 2);
    assert_eq!(search.encrypted_index, "opaque");
    assert_eq!(result.search_result_index, 3);
    assert_eq!(result.source, "kb");
}

#[rstest]
fn text_block_null_optionals_are_omitted() {
    let block: MessagesContentPart = serde_json::from_value(
        json!({"type":"text","text":"hi","citations":null,"cache_control":null,"future":null}),
    )
    .unwrap();
    let MessagesContentPart::Text(text) = &block else {
        panic!("expected text block");
    };
    assert!(text.citations.is_none());
    assert!(text.cache_control.is_none());
    assert_eq!(text.extra.get("future"), Some(&Value::Null));
    assert_eq!(
        serde_json::to_value(block).unwrap(),
        json!({"type":"text","text":"hi","future":null})
    );
}

#[rstest]
fn request_media_blocks_expose_sources() {
    let MessagesContentPart::Image(image) = part(json!({
        "type":"image",
        "source":{"type":"base64","media_type":"image/png","data":"AA=="},
        "transformations":{"oversized_image":"error"}
    })) else {
        panic!("expected image block");
    };
    assert!(
        matches!(&image.source, ContentSource::Base64 { media_type, .. } if media_type == "image/png")
    );
    assert_eq!(image.extra["transformations"]["oversized_image"], "error");
    let MessagesContentPart::Document(document) = part(json!({
        "type":"document",
        "source":{"type":"content","content":[
            {"type":"text","text":"Section 1"},
            {"type":"image","source":{"type":"url","url":"https://example.test/chart.png"}}
        ]},
        "title":"Q3 report",
        "context":"quarterly",
        "citations":{"enabled":true}
    })) else {
        panic!("expected document block");
    };
    assert_eq!(document.title.as_deref(), Some("Q3 report"));
    assert_eq!(document.citations.as_ref().unwrap().enabled, Some(true));
    let ContentSource::Content {
        content: BlockContent::Blocks(blocks),
        ..
    } = &document.source
    else {
        panic!("expected content-block source");
    };
    assert!(matches!(
        blocks.as_slice(),
        [MessagesContentPart::Text(_), MessagesContentPart::Image(_)]
    ));
    let MessagesContentPart::SearchResult(search) = part(json!({
        "type":"search_result",
        "source":"https://example.test/result",
        "title":"result",
        "content":[{"type":"text","text":"found"}],
        "citations":{"enabled":false}
    })) else {
        panic!("expected search result block");
    };
    assert_eq!(search.source, "https://example.test/result");
    assert!(
        matches!(search.content.as_slice(), [MessagesContentPart::Text(text)] if text.text == "found")
    );
}

#[rstest]
fn reasoning_and_tool_blocks_expose_required_fields() {
    let MessagesContentPart::Thinking(thinking) =
        part(json!({"type":"thinking","thinking":"plan","signature":"sig"}))
    else {
        panic!("expected thinking block");
    };
    assert_eq!(
        (thinking.thinking.as_str(), thinking.signature.as_str()),
        ("plan", "sig")
    );
    let MessagesContentPart::RedactedThinking(redacted) =
        part(json!({"type":"redacted_thinking","data":"opaque"}))
    else {
        panic!("expected redacted thinking block");
    };
    assert_eq!(redacted.data, "opaque");
    let MessagesContentPart::ToolUse(tool_use) = part(json!({
        "type":"tool_use",
        "id":"toolu_1",
        "name":"lookup",
        "input":{"query":[1,null]},
        "caller":{"type":"code_execution_20260120","tool_id":"srvtoolu_1"},
        "toolset_name":"browser"
    })) else {
        panic!("expected tool use block");
    };
    assert_eq!(tool_use.input["query"], json!([1, null]));
    assert!(
        matches!(&tool_use.caller, Some(ToolCaller::CodeExecution20260120 { tool_id, .. }) if tool_id == "srvtoolu_1")
    );
    assert_eq!(tool_use.toolset_name.as_deref(), Some("browser"));
    let MessagesContentPart::ToolResult(result) = part(json!({
        "type":"tool_result",
        "tool_use_id":"toolu_1",
        "is_error":false,
        "content":[
            {"type":"tool_reference","tool_name":"lookup"},
            {"type":"browser_state","tabs":[{"tab_id":"1","title":"","url":"","active":true}],
             "state_changes":[{"type":"download_completed","download_id":"d1","url":"https://example.test/f","size_bytes":3}]}
        ]
    })) else {
        panic!("expected tool result block");
    };
    let Some(BlockContent::Blocks(blocks)) = &result.content else {
        panic!("expected nested result blocks");
    };
    let [
        MessagesContentPart::ToolReference(reference),
        MessagesContentPart::BrowserState(browser),
    ] = blocks.as_slice()
    else {
        panic!("expected tool reference and browser state");
    };
    assert_eq!(reference.tool_name, "lookup");
    assert_eq!(browser.tabs[0].active, Some(true));
    assert!(matches!(
        browser.state_changes.as_deref(),
        Some([BrowserStateChange::DownloadCompleted {
            size_bytes: Some(3),
            path: None,
            ..
        }])
    ));
    let MessagesContentPart::ToolResult(text_result) =
        part(json!({"type":"tool_result","tool_use_id":"toolu_2","content":"done"}))
    else {
        panic!("expected tool result block");
    };
    assert_eq!(text_result.content, Some(BlockContent::Text("done".into())));
    assert!(text_result.is_error.is_none());
}

#[rstest]
fn server_tool_results_expose_nested_result_unions() {
    let MessagesContentPart::ServerToolUse(server) = part(
        json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"rust"}}),
    ) else {
        panic!("expected server tool use");
    };
    assert_eq!(server.name, "web_search");
    let MessagesContentPart::WebSearchToolResult(search) = part(json!({
        "type":"web_search_tool_result",
        "tool_use_id":"srvtoolu_1",
        "content":[{"type":"web_search_result","url":"https://example.test","title":"t","encrypted_content":"e","page_age":"1d"}]
    })) else {
        panic!("expected web search result");
    };
    let WebSearchToolResultContent::Results(results) = &search.content else {
        panic!("expected search results");
    };
    assert_eq!(results[0].page_age.as_deref(), Some("1d"));
    let MessagesContentPart::WebSearchToolResult(search_error) = part(json!({
        "type":"web_search_tool_result",
        "tool_use_id":"srvtoolu_1",
        "content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}
    })) else {
        panic!("expected web search error");
    };
    assert!(
        matches!(&search_error.content, WebSearchToolResultContent::Error(error) if error.error_code == "max_uses_exceeded")
    );
    let MessagesContentPart::WebFetchToolResult(fetch) = part(json!({
        "type":"web_fetch_tool_result",
        "tool_use_id":"srvtoolu_2",
        "content":{"type":"web_fetch_result","url":"https://example.test","retrieved_at":"2026-01-01T00:00:00Z",
            "content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"page"}}}
    })) else {
        panic!("expected web fetch result");
    };
    let WebFetchToolResultContent::WebFetchResult(fetched) = &fetch.content else {
        panic!("expected fetched page");
    };
    assert!(matches!(
        fetched.content.as_ref(),
        MessagesContentPart::Document(_)
    ));
    let MessagesContentPart::CodeExecutionToolResult(code) = part(json!({
        "type":"code_execution_tool_result",
        "tool_use_id":"srvtoolu_3",
        "content":{"type":"encrypted_code_execution_result","encrypted_stdout":"opaque","stderr":"","return_code":-1,
            "content":[{"type":"code_execution_output","file_id":"file_1"}]}
    })) else {
        panic!("expected code execution result");
    };
    let CodeExecutionToolResultContent::EncryptedCodeExecutionResult(encrypted) = &code.content
    else {
        panic!("expected encrypted execution result");
    };
    assert_eq!(encrypted.return_code, -1);
    assert!(
        matches!(encrypted.content.as_slice(), [CodeExecutionOutput::CodeExecutionOutput { file_id, .. }] if file_id == "file_1")
    );
    let MessagesContentPart::TextEditorCodeExecutionToolResult(editor) = part(json!({
        "type":"text_editor_code_execution_tool_result",
        "tool_use_id":"srvtoolu_4",
        "content":{"type":"text_editor_code_execution_view_result","content":"fn main() {}","file_type":"text","num_lines":1}
    })) else {
        panic!("expected text editor result");
    };
    let TextEditorCodeExecutionToolResultContent::TextEditorCodeExecutionViewResult(view) =
        &editor.content
    else {
        panic!("expected view result");
    };
    assert_eq!(view.file_type, TextEditorFileType::Text);
    assert_eq!(view.num_lines, Some(1));
    assert!(view.start_line.is_none());
    let MessagesContentPart::ToolSearchToolResult(tool_search) = part(json!({
        "type":"tool_search_tool_result",
        "tool_use_id":"srvtoolu_5",
        "content":{"type":"tool_search_tool_result_error","error_code":"unavailable","error_message":"down"}
    })) else {
        panic!("expected tool search result");
    };
    assert!(
        matches!(&tool_search.content, ToolSearchToolResultContent::ToolSearchToolResultError(error) if error.error_message.as_deref() == Some("down"))
    );
    let MessagesContentPart::AdvisorToolResult(advisor) = part(json!({
        "type":"advisor_tool_result",
        "tool_use_id":"srvtoolu_6",
        "content":{"type":"advisor_result","text":"advice"}
    })) else {
        panic!("expected advisor result");
    };
    assert!(
        matches!(&advisor.content, AdvisorToolResultContent::AdvisorResult { text, stop_reason: None, .. } if text == "advice")
    );
}

#[rstest]
fn beta_blocks_expose_mcp_compaction_and_fallback_fields() {
    let MessagesContentPart::McpToolUse(mcp) = part(
        json!({"type":"mcp_tool_use","id":"mcptoolu_1","name":"search","server_name":"kb","input":{}}),
    ) else {
        panic!("expected MCP tool use");
    };
    assert_eq!(mcp.server_name, "kb");
    let MessagesContentPart::McpToolResult(mcp_result) = part(
        json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":true,"content":"failed"}),
    ) else {
        panic!("expected MCP tool result");
    };
    assert_eq!(mcp_result.is_error, Some(true));
    let MessagesContentPart::McpToolListing(listing) = part(json!({
        "type":"mcp_tool_listing",
        "mcp_server_name":"kb",
        "tools":[{"name":"search","input_schema":{"type":"object"}}]
    })) else {
        panic!("expected MCP tool listing");
    };
    assert!(listing.tools[0].description.is_none());
    let MessagesContentPart::Compaction(compaction) = part(json!({
        "type":"compaction",
        "content":"summary",
        "encrypted_content":"opaque",
        "tool_changes":[
            {"type":"tool_addition","tool":{"type":"tool_definition","definition":{"name":"lookup","input_schema":{"type":"object"}}}},
            {"type":"tool_removal","tool":{"type":"mcp_tool_reference","server_name":"kb","name":"search"}}
        ]
    })) else {
        panic!("expected compaction block");
    };
    let [
        ToolChange::ToolAddition(addition),
        ToolChange::ToolRemoval(removal),
    ] = compaction.tool_changes.as_deref().unwrap()
    else {
        panic!("expected tool addition and removal");
    };
    let ToolChangeTarget::ToolDefinition { definition, .. } = &addition.tool else {
        panic!("expected inline tool definition");
    };
    assert!(
        matches!(definition.as_ref(), MessagesToolParam::Custom(tool) if tool.name == "lookup")
    );
    assert!(
        matches!(&removal.tool, ToolChangeTarget::McpToolReference { server_name, .. } if server_name == "kb")
    );
    let failed: MessagesContentPart =
        serde_json::from_value(json!({"type":"compaction","content":null})).unwrap();
    assert_eq!(failed, MessagesContentPart::Compaction(Default::default()));
    let MessagesContentPart::Fallback(fallback) = part(json!({
        "type":"fallback",
        "from":{"model":"claude-opus-5-5"},
        "to":{"model":"claude-sonnet-5-5"},
        "trigger":{"type":"refusal","category":"cyber"}
    })) else {
        panic!("expected fallback block");
    };
    assert_ne!(fallback.from.model, fallback.to.model);
}

#[rstest]
#[case::missing_tag(json!({"text":"hello"}))]
#[case::unknown_tag(json!({"type":"future_block","text":"hello"}))]
#[case::text_without_text(json!({"type":"text"}))]
#[case::wrong_text(json!({"type":"text","text":7}))]
#[case::citation_missing_location(json!({"type":"text","text":"a","citations":[{"type":"char_location","cited_text":"a","document_index":0}]}))]
#[case::unknown_citation(json!({"type":"text","text":"a","citations":[{"type":"future_location"}]}))]
#[case::thinking_without_signature(json!({"type":"thinking","thinking":"plan"}))]
#[case::tool_use_without_id(json!({"type":"tool_use","name":"lookup","input":{}}))]
#[case::tool_use_array_input(json!({"type":"tool_use","id":"t","name":"lookup","input":[]}))]
#[case::tool_result_without_id(json!({"type":"tool_result","content":"done"}))]
#[case::malformed_nested_block(json!({"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":7}]}))]
#[case::malformed_source(json!({"type":"document","source":{"type":"url","url":7}}))]
#[case::image_without_source(json!({"type":"image"}))]
#[case::search_result_without_title(json!({"type":"search_result","source":"s","content":[]}))]
#[case::web_search_result_without_url(json!({"type":"web_search_tool_result","tool_use_id":"t","content":[{"type":"web_search_result","title":"t","encrypted_content":"e"}]}))]
#[case::unknown_fetch_result(json!({"type":"web_fetch_tool_result","tool_use_id":"t","content":{"type":"future"}}))]
#[case::fractional_return_code(json!({"type":"bash_code_execution_tool_result","tool_use_id":"t","content":{"type":"bash_code_execution_result","stdout":"","stderr":"","return_code":0.5,"content":[]}}))]
#[case::unknown_file_type(json!({"type":"text_editor_code_execution_tool_result","tool_use_id":"t","content":{"type":"text_editor_code_execution_view_result","content":"","file_type":"video"}}))]
#[case::negative_line_count(json!({"type":"text_editor_code_execution_tool_result","tool_use_id":"t","content":{"type":"text_editor_code_execution_str_replace_result","new_lines":-1}}))]
#[case::unknown_tool_change(json!({"type":"compaction","tool_changes":[{"type":"tool_addition","tool":{"type":"future"}}]}))]
#[case::unknown_state_change(json!({"type":"browser_state","tabs":[],"state_changes":[{"type":"tab_closed","tab_id":"1"}]}))]
fn content_parts_reject_malformed_known_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<MessagesContentPart>(wire).is_err());
}

#[rstest]
fn custom_tool_exposes_schema_and_preserves_extensions() {
    let tool = round_trip::<CustomTool>(json!({
        "type":"custom",
        "name":"lookup",
        "input_schema":{"type":"object","properties":{"query":{"type":"string"}}},
        "strict":false,
        "defer_loading":true,
        "extension":{"nested":[1,null]}
    }));
    assert_eq!(tool.tool_type, Some(CustomToolType::Custom));
    assert_eq!(tool.name, "lookup");
    assert_eq!((tool.strict, tool.defer_loading), (Some(false), Some(true)));
    let litellm_llms_types::json_schema::JsonSchema::Object(schema) = &tool.input_schema else {
        panic!("expected an object schema");
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("query"));
    assert_eq!(tool.extra["extension"], json!({"nested":[1,null]}));
}

#[rstest]
fn custom_tool_omits_null_discriminator() {
    let tool: CustomTool =
        serde_json::from_value(json!({"type":null,"name":"lookup","input_schema":true})).unwrap();
    assert!(tool.tool_type.is_none());
    assert!(tool.description.is_none());
    assert_eq!(
        serde_json::to_value(tool).unwrap(),
        json!({"name":"lookup","input_schema":true})
    );
}

#[rstest]
#[case::missing_name(json!({"input_schema":{}}))]
#[case::missing_schema(json!({"name":"lookup"}))]
#[case::wrong_name_shape(json!({"name":7,"input_schema":{}}))]
#[case::builtin_tool(json!({"name":"lookup","input_schema":{},"type":"bash_20250124"}))]
#[case::unknown_discriminator(json!({"name":"lookup","input_schema":{},"type":"future_tool"}))]
fn custom_tool_rejects_invalid_shapes(#[case] wire: Value) {
    assert!(serde_json::from_value::<CustomTool>(wire).is_err());
}

#[rstest]
#[case::builtin(json!({"type":"web_search_20250305","name":"web_search","max_uses":2}), true)]
#[case::custom(json!({"name":"lookup","input_schema":{"type":"object"}}), false)]
fn tool_params_dispatch_on_discriminator(#[case] wire: Value, #[case] builtin: bool) {
    let tool = round_trip::<MessagesToolParam>(wire);
    assert_eq!(matches!(tool, MessagesToolParam::Builtin(_)), builtin);
}

#[rstest]
fn tool_params_reject_unknown_tool_types() {
    assert!(
        serde_json::from_value::<MessagesToolParam>(
            json!({"type":"future_tool","name":"lookup","input_schema":{}})
        )
        .is_err()
    );
}

#[rstest]
fn prompt_cache_breakpoint_round_trips() {
    let breakpoint = round_trip::<PromptCacheBreakpoint>(json!({"mode":"explicit"}));
    assert!(breakpoint.mode.is_some());
}
