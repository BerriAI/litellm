use litellm_llms_types::formats::messages::{
    AppliedEdit, BlockContent, BuiltinMessagesTool, Citation, Citations, ContainerReference,
    ContentBlockPayload, ContentBlockSource, ContentSource, ContextManagementResponse,
    ContextTrigger, CustomTool, CustomToolType, McpServer, MessageRole, MessageType,
    MessagesCompaction, MessagesContainer, MessagesContentBlock, MessagesMetadata, MessagesUsage,
    OutputFormat, PromptCacheBreakpoint, Safeguard, StopDetails, StopReason, ToolCaller,
    ToolChoice, ToolDefinition, WebSearchResultError,
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
        "speed":"fast"
    }));
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
    let citations = round_trip::<Citations>(wire);
    match citations {
        Citations::Config(config) => {
            assert_eq!(config.enabled, Some(true));
            assert_eq!(config.extra.get("future"), Some(&Value::Null));
        }
        Citations::Results(results) => {
            let [citation] = results.as_slice() else {
                panic!("expected one citation");
            };
            match citation {
                Citation::PageLocation(page) => {
                    assert_eq!(page.cited_text.as_deref(), Some("quote"));
                    assert_eq!(page.start_page_number, Some(1));
                }
                Citation::CharLocation(chars) => {
                    assert_eq!(chars.start_char_index, Some(1));
                    assert_eq!(chars.end_char_index, Some(6));
                }
                Citation::WebSearchResultLocation(search) => {
                    assert_eq!(search.url.as_deref(), Some("https://example.test"));
                    assert_eq!(search.title.as_deref(), Some("result"));
                }
                Citation::ContentBlockLocation(block) => {
                    assert_eq!(block.start_block_index, Some(1));
                    assert_eq!(block.end_block_index, Some(2));
                }
                Citation::SearchResultLocation(search) => {
                    assert_eq!(search.search_result_index, Some(0));
                    assert_eq!(search.source.as_deref(), Some("web"));
                }
            }
        }
    }
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
#[case::builtin_tool(json!({"name":"lookup","type":"bash_20250124"}))]
#[case::unknown_discriminator(json!({"name":"lookup","type":"future_tool"}))]
#[case::wrong_discriminator_shape(json!({"name":"lookup","type":7}))]
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
    let Some(ContentBlockSource::Source(ContentSource::Url { url, .. })) = &payload.source else {
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

#[rstest]
#[case::absent(json!({"name":"lookup"}))]
#[case::null(json!({"name":"lookup","type":null}))]
#[case::custom(json!({"name":"lookup","type":"custom"}))]
fn custom_tool_accepts_optional_discriminator(#[case] wire: Value) {
    let tool: CustomTool = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(tool.definition.name.as_deref(), Some("lookup"));
    assert!(tool.definition.description.is_none());
    assert!(tool.definition.input_schema.is_none());
    assert!(tool.definition.strict.is_none());
    assert_eq!(
        tool.tool_type,
        wire.get("type")
            .and_then(Value::as_str)
            .map(|_| CustomToolType::Custom)
    );
    assert!(!tool.definition.extra.contains_key("type"));
    let expected = if wire.get("type") == Some(&Value::Null) {
        json!({"name":"lookup"})
    } else {
        wire
    };
    assert_eq!(serde_json::to_value(tool).unwrap(), expected);
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
#[case::missing(json!({"extension":null}))]
#[case::null(json!({"text":null,"content":null,"source":null,"citations":null,"return_code":null,"extension":null}))]
fn content_payload_optional_fields_are_omitted(#[case] wire: Value) {
    let payload: ContentBlockPayload = serde_json::from_value(wire).unwrap();
    assert!(payload.text.is_none());
    assert!(payload.content.is_none());
    assert!(payload.source.is_none());
    assert!(payload.citations.is_none());
    assert!(payload.return_code.is_none());
    assert_eq!(payload.extra.get("extension"), Some(&Value::Null));
    assert_eq!(
        serde_json::to_value(payload).unwrap(),
        json!({"extension":null})
    );
}

#[rstest]
#[case::wrong_text(json!({"text":7}))]
#[case::wrong_input(json!({"input":[]}))]
#[case::wrong_content(json!({"content":7}))]
#[case::malformed_source(json!({"source":{"url":"https://example.test"}}))]
#[case::malformed_caller(json!({"caller":{"type":"code_execution_20250825"}}))]
#[case::wrong_citations(json!({"citations":7}))]
#[case::negative_lines(json!({"num_lines":-1}))]
#[case::wrong_error_flag(json!({"is_error":"false"}))]
fn content_payload_rejects_malformed_known_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ContentBlockPayload>(wire).is_err());
}

#[rstest]
fn search_results_expose_typed_location_and_nested_blocks() {
    let wire = json!({
        "type":"search_result",
        "source":"https://example.test/result",
        "title":"result",
        "content":[{"type":"text","text":"found","extension":null}],
        "citations":{"enabled":true},
        "extension":{"nested":[1,null]}
    });
    let block: MessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    let MessagesContentBlock::SearchResult(payload) = &block else {
        panic!("expected search result");
    };
    let Some(ContentBlockSource::Location(location)) = &payload.source else {
        panic!("expected search location");
    };
    assert_eq!(location, "https://example.test/result");
    let Some(BlockContent::Blocks(blocks)) = &payload.content else {
        panic!("expected nested blocks");
    };
    let [MessagesContentBlock::Text(text)] = blocks.as_slice() else {
        panic!("expected text block");
    };
    assert_eq!(text.text.as_deref(), Some("found"));
    assert_eq!(text.extra.get("extension"), Some(&Value::Null));
    assert_eq!(serde_json::to_value(block).unwrap(), wire);
}

#[rstest]
fn content_payload_exposes_recursive_tool_contracts() {
    let wire = json!({
        "tool":{"type":"tool_use","name":"lookup","input":{"query":[1,null]}},
        "tool_references":[{"type":"tool_reference","tool_name":"lookup"}],
        "content":{"type":"code_execution_result","stdout":"done","return_code":0},
        "caller":{"type":"code_execution_20260120","tool_id":"call_1"}
    });
    let payload: ContentBlockPayload = serde_json::from_value(wire.clone()).unwrap();
    let Some(MessagesContentBlock::ToolUse(tool)) = payload.tool.as_deref() else {
        panic!("expected tool use");
    };
    assert_eq!(tool.name.as_deref(), Some("lookup"));
    assert_eq!(
        tool.input.as_ref().unwrap().get("query"),
        Some(&json!([1, null]))
    );
    let Some(references) = &payload.tool_references else {
        panic!("expected tool references");
    };
    let [MessagesContentBlock::ToolReference(reference)] = references.as_slice() else {
        panic!("expected typed tool reference");
    };
    assert_eq!(reference.tool_name.as_deref(), Some("lookup"));
    let Some(BlockContent::Block(result)) = &payload.content else {
        panic!("expected single result block");
    };
    let MessagesContentBlock::CodeExecutionResult(result_payload) = result.as_ref() else {
        panic!("expected execution result");
    };
    assert_eq!(result_payload.stdout.as_deref(), Some("done"));
    assert_eq!(result_payload.return_code, Some(0));
    let Some(ToolCaller::CodeExecution20260120 { tool_id, .. }) = &payload.caller else {
        panic!("expected versioned tool caller");
    };
    assert_eq!(tool_id, "call_1");
    assert_eq!(serde_json::to_value(payload).unwrap(), wire);
}

#[rstest]
#[case::missing_tag(json!({"text":"hello"}))]
#[case::unknown_tag(json!({"type":"future_block","text":"hello"}))]
#[case::wrong_text(json!({"type":"text","text":7}))]
#[case::malformed_recursive_tool(json!({"type":"tool_use","tool":{"name":"lookup"}}))]
#[case::malformed_recursive_content(json!({"type":"tool_result","content":[{"type":"text","text":7}]}))]
#[case::malformed_source(json!({"type":"document","source":{"type":"url","url":7}}))]
fn typed_content_blocks_reject_malformed_known_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<MessagesContentBlock>(wire).is_err());
}

#[rstest]
fn typed_content_blocks_accept_partial_payloads() {
    let wire = json!({"type":"text","extension":null});
    let block: MessagesContentBlock = serde_json::from_value(wire.clone()).unwrap();
    let MessagesContentBlock::Text(payload) = &block else {
        panic!("expected text block");
    };
    assert!(payload.text.is_none());
    assert_eq!(payload.extra.get("extension"), Some(&Value::Null));
    assert_eq!(serde_json::to_value(block).unwrap(), wire);
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
