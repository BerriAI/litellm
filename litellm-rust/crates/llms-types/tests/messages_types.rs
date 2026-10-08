use litellm_llms_types::formats::messages::{
    AdvisorToolResultContent, AllowedCaller, BashCodeExecutionOutput,
    BashCodeExecutionToolResultContent, BlockContent, BrowserStateChange, BuiltinMessagesTool,
    CacheMissReason, Citation, CodeExecutionOutput, CodeExecutionToolResultContent,
    ContainerReference, ContentSource, ContextManagementResponse, ContextTrigger, CustomTool,
    CustomToolType, FallbackTrigger, McpServer, McpToolResultContent, McpToolResultText,
    MessageRole, MessageType, MessagesCompaction, MessagesContainer, MessagesContentPart,
    MessagesDiagnostics, MessagesDiagnosticsParam, MessagesMetadata, MessagesToolParam,
    MessagesUsage, OutputFormat, ResponseInclusion, Safeguard, SkillType, StopDetails,
    StopDetailsType, StopReason, TextEditorCodeExecutionToolResultContent, TextEditorFileType,
    ToolCaller, ToolChange, ToolChangeTarget, ToolChoice, ToolChoiceType, ToolResultUrlSource,
    ToolSearchReference, ToolSearchToolResultContent, UrlSourceToolReference, UsageIterationType,
    UserInputUrlSource, WebFetchDocument, WebFetchToolResultContent, WebSearchToolResultContent,
};
use litellm_llms_types::json_schema::JsonSchema;
use litellm_llms_types::recognized::Recognized;
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
    assert!(output_format.extra.is_empty());
    let JsonSchema::Object(schema) = &output_format.schema else {
        panic!("expected output object schema");
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("name"));
    let compaction =
        round_trip::<MessagesCompaction>(json!({"type":"summarize","instructions":"briefly"}));
    assert_eq!(compaction.instructions.as_deref(), Some("briefly"));
    assert!(compaction.extra.is_empty());
    let container = round_trip::<MessagesContainer>(json!({
        "id":"container_1",
        "expires_at":"2026-01-01T00:00:00Z",
        "skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]
    }));
    assert_eq!(container.id.as_deref(), Some("container_1"));
    assert!(container.extra.is_empty());
    let [skill] = container.skills.as_ref().unwrap().as_slice() else {
        panic!("expected container skill");
    };
    assert_eq!(skill.skill_type, SkillType::Custom);
    assert_eq!(skill.skill_id, "skill_1");
    assert_eq!(skill.version.as_deref(), Some("1"));
    assert!(skill.extra.is_empty());
    let reference = round_trip::<ContainerReference>(json!({"id":"container_1"}));
    let ContainerReference::Parameters(parameters) = reference else {
        panic!("expected container parameters");
    };
    assert_eq!(parameters.id.as_deref(), container.id.as_deref());
    assert!(parameters.skills.is_none());
    round_trip::<ContainerReference>(
        json!({"id":"container_1","skills":[{"type":"anthropic","skill_id":"pptx"}]}),
    );
    let server = round_trip::<McpServer>(json!({
        "type":"url",
        "url":"https://example.test/mcp",
        "name":"search",
        "authorization_token":"token",
        "tool_configuration":{"allowed_tools":["search"],"enabled":true}
    }));
    assert_eq!(server.url, "https://example.test/mcp");
    assert_eq!(server.name, "search");
    assert_eq!(server.authorization_token.as_deref(), Some("token"));
    assert!(server.extra.is_empty());
    let configuration = server.tool_configuration.as_ref().unwrap();
    assert_eq!(configuration.enabled, Some(true));
    assert_eq!(
        configuration.allowed_tools.as_deref(),
        Some([String::from("search")].as_slice())
    );
    assert!(configuration.extra.is_empty());
    let context = round_trip::<ContextManagementResponse>(json!({
        "applied_edits":[
            {"type":"clear_tool_uses_20250919","cleared_input_tokens":7,"cleared_tool_uses":2},
            {"type":"clear_thinking_20251015","cleared_input_tokens":3,"cleared_thinking_turns":1}
        ]
    }));
    let [tool_uses, thinking] = context.applied_edits.as_ref().unwrap().as_slice() else {
        panic!("expected two applied edits");
    };
    assert_eq!(
        tool_uses.edit_type.as_deref(),
        Some("clear_tool_uses_20250919")
    );
    assert_eq!(
        (tool_uses.cleared_input_tokens, tool_uses.cleared_tool_uses),
        (Some(7), Some(2))
    );
    assert!(tool_uses.cleared_thinking_turns.is_none());
    assert_eq!(
        (
            thinking.cleared_input_tokens,
            thinking.cleared_thinking_turns
        ),
        (Some(3), Some(1))
    );
    assert!(tool_uses.extra.is_empty() && thinking.extra.is_empty());
    let safeguard = round_trip::<Safeguard>(
        json!({"type":"classifier","classifier_context":{"source":"test"}}),
    );
    assert_eq!(safeguard.safeguard_type, "classifier");
    assert_eq!(
        safeguard.classifier_context.as_ref().unwrap().get("source"),
        Some(&json!("test"))
    );
    assert!(safeguard.extra.is_empty());
}

#[rstest]
fn stop_details_expose_refusal_and_keep_unknown_types() {
    let refusal = round_trip::<StopDetails>(
        json!({"type":"refusal","category":"cyber","explanation":"blocked"}),
    );
    assert_eq!(
        refusal.detail_type,
        Recognized::Known(StopDetailsType::Refusal)
    );
    assert_eq!(refusal.category.as_deref(), Some("cyber"));
    assert_eq!(refusal.explanation.as_deref(), Some("blocked"));
    assert!(refusal.extra.is_empty());
    let safeguard =
        round_trip::<StopDetails>(json!({"type":"safeguard","safeguard_types":["classifier"]}));
    assert_eq!(
        safeguard.detail_type,
        Recognized::Unrecognized(json!("safeguard"))
    );
    assert_eq!(safeguard.extra["safeguard_types"], json!(["classifier"]));
}

#[rstest]
#[case::skill_without_id(
    json!({"skills":[{"type":"custom","version":"1"}]}),
    |wire| serde_json::from_value::<MessagesContainer>(wire).is_err()
)]
#[case::mcp_server_without_url(
    json!({"type":"url","name":"search"}),
    |wire| serde_json::from_value::<McpServer>(wire).is_err()
)]
#[case::mcp_server_without_name(
    json!({"type":"url","url":"https://example.test/mcp"}),
    |wire| serde_json::from_value::<McpServer>(wire).is_err()
)]
#[case::output_format_without_schema(
    json!({"type":"json_schema"}),
    |wire| serde_json::from_value::<OutputFormat>(wire).is_err()
)]
fn metadata_contracts_reject_missing_required_fields(
    #[case] wire: Value,
    #[case] rejects: fn(Value) -> bool,
) {
    assert!(rejects(wire));
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
fn tool_choice_round_trips() {
    let choice = round_trip::<ToolChoice>(
        json!({"type":"tool","name":"search","disable_parallel_tool_use":true}),
    );
    assert_eq!(choice.choice_type, ToolChoiceType::Tool);
    assert_eq!(choice.name.as_deref(), Some("search"));
    assert_eq!(choice.disable_parallel_tool_use, Some(true));
    assert!(choice.extra.is_empty());
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
    assert!(usage.extra.is_empty());
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
        Recognized::Known(UsageIterationType::Compaction)
    );
    assert_eq!(
        message.iteration_type,
        Recognized::Known(UsageIterationType::Message)
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
fn usage_iterations_expose_advisor_and_fallback_models() {
    let usage = round_trip::<MessagesUsage>(json!({
        "iterations":[
            {"type":"advisor_message","model":"claude-opus-5-5","input_tokens":5,"output_tokens":2,
             "cache_creation_input_tokens":4,"cache_read_input_tokens":1,
             "cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":4}},
            {"type":"fallback_message","model":"claude-sonnet-5-5","input_tokens":6,"output_tokens":3,
             "cache_creation_input_tokens":0,"cache_read_input_tokens":2},
            {"type":"future_iteration","input_tokens":1}
        ]
    }));
    let [advisor, fallback, future] = usage.iterations.as_deref().unwrap() else {
        panic!("expected three iterations");
    };
    assert_eq!(
        advisor.iteration_type,
        Recognized::Known(UsageIterationType::AdvisorMessage)
    );
    assert_eq!(advisor.model.as_deref(), Some("claude-opus-5-5"));
    assert_eq!(
        (
            advisor.cache_creation_input_tokens,
            advisor.cache_read_input_tokens
        ),
        (Some(4), Some(1))
    );
    assert_eq!(
        advisor
            .cache_creation
            .as_ref()
            .unwrap()
            .ephemeral_5m_input_tokens,
        advisor.cache_creation_input_tokens
    );
    assert!(advisor.extra.is_empty());
    assert_eq!(
        fallback.iteration_type,
        Recognized::Known(UsageIterationType::FallbackMessage)
    );
    assert_eq!(fallback.model.as_deref(), Some("claude-sonnet-5-5"));
    assert_eq!(fallback.cache_read_input_tokens, Some(2));
    assert!(fallback.extra.is_empty());
    assert_eq!(
        future.iteration_type,
        Recognized::Unrecognized(json!("future_iteration"))
    );
    assert_eq!(future.input_tokens, Some(1));
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
#[case::bash_20241022(
    json!({"type":"bash_20241022","name":"bash","input_examples":[{"command":"ls"}]}),
    |tool| {
        let BuiltinMessagesTool::Bash20241022(bash) = tool else { panic!("expected bash_20241022") };
        assert_eq!(bash.input_examples.unwrap()[0]["command"], "ls");
        assert!(bash.extra.is_empty());
    }
)]
#[case::bash_20250124(
    json!({"type":"bash_20250124","name":"bash","allowed_callers":["direct","code_execution_20260521"]}),
    |tool| {
        let BuiltinMessagesTool::Bash20250124(bash) = tool else { panic!("expected bash_20250124") };
        assert_eq!(
            bash.allowed_callers.as_deref(),
            Some([AllowedCaller::Direct, AllowedCaller::CodeExecution20260521].as_slice())
        );
        assert!(bash.extra.is_empty());
    }
)]
#[case::text_editor_20241022(
    json!({"type":"text_editor_20241022","name":"str_replace_editor","defer_loading":true}),
    |tool| {
        let BuiltinMessagesTool::TextEditor20241022(editor) = tool else { panic!("expected text_editor_20241022") };
        assert_eq!(editor.defer_loading, Some(true));
        assert!(editor.extra.is_empty());
    }
)]
#[case::text_editor_20250124(
    json!({"type":"text_editor_20250124","name":"str_replace_editor","strict":true}),
    |tool| {
        let BuiltinMessagesTool::TextEditor20250124(editor) = tool else { panic!("expected text_editor_20250124") };
        assert_eq!(editor.strict, Some(true));
        assert!(editor.extra.is_empty());
    }
)]
#[case::text_editor_20250429(
    json!({"type":"text_editor_20250429","name":"str_replace_based_edit_tool","cache_control":{"type":"ephemeral","ttl":"1h"}}),
    |tool| {
        let BuiltinMessagesTool::TextEditor20250429(editor) = tool else { panic!("expected text_editor_20250429") };
        assert_eq!(editor.cache_control.unwrap().ttl.as_deref(), Some("1h"));
        assert!(editor.extra.is_empty());
    }
)]
#[case::text_editor_20250728(
    json!({"type":"text_editor_20250728","name":"str_replace_based_edit_tool","max_characters":10000}),
    |tool| {
        let BuiltinMessagesTool::TextEditor20250728(editor) = tool else { panic!("expected text_editor_20250728") };
        assert_eq!(editor.max_characters, Some(10000));
        assert!(editor.extra.is_empty());
    }
)]
#[case::memory_20250818(
    json!({"type":"memory_20250818","name":"memory","allowed_callers":["code_execution_20250825"]}),
    |tool| {
        let BuiltinMessagesTool::Memory20250818(memory) = tool else { panic!("expected memory_20250818") };
        assert_eq!(memory.allowed_callers, Some(vec![AllowedCaller::CodeExecution20250825]));
        assert!(memory.extra.is_empty());
    }
)]
#[case::computer_20241022(
    json!({"type":"computer_20241022","name":"computer","display_width_px":1024,"display_height_px":768,"display_number":1}),
    |tool| {
        let BuiltinMessagesTool::Computer20241022(computer) = tool else { panic!("expected computer_20241022") };
        assert_eq!((computer.display_width_px, computer.display_height_px), (1024, 768));
        assert_eq!(computer.display_number, Some(1));
        assert!(computer.extra.is_empty());
    }
)]
#[case::computer_20250124(
    json!({"type":"computer_20250124","name":"computer","display_width_px":1280,"display_height_px":800}),
    |tool| {
        let BuiltinMessagesTool::Computer20250124(computer) = tool else { panic!("expected computer_20250124") };
        assert_eq!((computer.display_width_px, computer.display_height_px), (1280, 800));
        assert!(computer.display_number.is_none());
        assert!(computer.extra.is_empty());
    }
)]
#[case::computer_20251124(
    json!({"type":"computer_20251124","name":"computer","display_width_px":1920,"display_height_px":1080,"enable_zoom":true}),
    |tool| {
        let BuiltinMessagesTool::Computer20251124(computer) = tool else { panic!("expected computer_20251124") };
        assert_eq!((computer.display_width_px, computer.display_height_px), (1920, 1080));
        assert_eq!(computer.enable_zoom, Some(true));
        assert!(computer.extra.is_empty());
    }
)]
#[case::code_execution_20250522(
    json!({"type":"code_execution_20250522","name":"code_execution","strict":false}),
    |tool| {
        let BuiltinMessagesTool::CodeExecution20250522(code) = tool else { panic!("expected code_execution_20250522") };
        assert_eq!(code.strict, Some(false));
        assert!(code.extra.is_empty());
    }
)]
#[case::code_execution_20250825(
    json!({"type":"code_execution_20250825","name":"code_execution","defer_loading":false}),
    |tool| {
        let BuiltinMessagesTool::CodeExecution20250825(code) = tool else { panic!("expected code_execution_20250825") };
        assert_eq!(code.defer_loading, Some(false));
        assert!(code.extra.is_empty());
    }
)]
#[case::code_execution_20260120(
    json!({"type":"code_execution_20260120","name":"code_execution","allowed_callers":["direct"]}),
    |tool| {
        let BuiltinMessagesTool::CodeExecution20260120(code) = tool else { panic!("expected code_execution_20260120") };
        assert_eq!(code.allowed_callers, Some(vec![AllowedCaller::Direct]));
        assert!(code.extra.is_empty());
    }
)]
#[case::code_execution_20260521(
    json!({"type":"code_execution_20260521","name":"code_execution","allowed_callers":["code_execution_20260120"]}),
    |tool| {
        let BuiltinMessagesTool::CodeExecution20260521(code) = tool else { panic!("expected code_execution_20260521") };
        assert_eq!(code.allowed_callers, Some(vec![AllowedCaller::CodeExecution20260120]));
        assert!(code.extra.is_empty());
    }
)]
#[case::tool_search_regex_20251119(
    json!({"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex","defer_loading":true}),
    |tool| {
        let BuiltinMessagesTool::ToolSearchRegex20251119(search) = tool else { panic!("expected tool_search_tool_regex_20251119") };
        assert_eq!(search.defer_loading, Some(true));
        assert!(search.extra.is_empty());
    }
)]
#[case::tool_search_regex(
    json!({"type":"tool_search_tool_regex","name":"tool_search_tool_regex","strict":true}),
    |tool| {
        let BuiltinMessagesTool::ToolSearchRegex(search) = tool else { panic!("expected tool_search_tool_regex") };
        assert_eq!(search.strict, Some(true));
        assert!(search.extra.is_empty());
    }
)]
#[case::tool_search_bm25_20251119(
    json!({"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_bm25","defer_loading":false}),
    |tool| {
        let BuiltinMessagesTool::ToolSearchBm2520251119(search) = tool else { panic!("expected tool_search_tool_bm25_20251119") };
        assert_eq!(search.defer_loading, Some(false));
        assert!(search.extra.is_empty());
    }
)]
#[case::tool_search_bm25(
    json!({"type":"tool_search_tool_bm25","name":"tool_search_tool_bm25","strict":false}),
    |tool| {
        let BuiltinMessagesTool::ToolSearchBm25(search) = tool else { panic!("expected tool_search_tool_bm25") };
        assert_eq!(search.strict, Some(false));
        assert!(search.extra.is_empty());
    }
)]
#[case::web_search_20250305(
    json!({"type":"web_search_20250305","name":"web_search","max_uses":3,"allowed_domains":["example.test"],
        "user_location":{"type":"approximate","city":"San Francisco","country":"US","timezone":"America/Los_Angeles"}}),
    |tool| {
        let BuiltinMessagesTool::WebSearch20250305(search) = tool else { panic!("expected web_search_20250305") };
        assert_eq!(search.max_uses, Some(3));
        assert_eq!(search.allowed_domains, Some(vec![String::from("example.test")]));
        let location = search.user_location.unwrap();
        assert_eq!(location.city.as_deref(), Some("San Francisco"));
        assert_eq!(location.timezone.as_deref(), Some("America/Los_Angeles"));
        assert!(location.extra.is_empty());
        assert!(search.extra.is_empty());
    }
)]
#[case::web_search_20260209(
    json!({"type":"web_search_20260209","name":"web_search","blocked_domains":["blocked.test"]}),
    |tool| {
        let BuiltinMessagesTool::WebSearch20260209(search) = tool else { panic!("expected web_search_20260209") };
        assert_eq!(search.blocked_domains, Some(vec![String::from("blocked.test")]));
        assert!(search.extra.is_empty());
    }
)]
#[case::web_search_20260318(
    json!({"type":"web_search_20260318","name":"web_search","response_inclusion":"excluded"}),
    |tool| {
        let BuiltinMessagesTool::WebSearch20260318(search) = tool else { panic!("expected web_search_20260318") };
        assert_eq!(search.response_inclusion, Some(ResponseInclusion::Excluded));
        assert!(search.extra.is_empty());
    }
)]
#[case::web_fetch_20250910(
    json!({"type":"web_fetch_20250910","name":"web_fetch","max_content_tokens":5000,"citations":{"enabled":true},
        "url_sources":{"client_tool_results":{"type":"only","tools":[{"type":"tool_reference","name":"lookup"}]},
            "server_tool_results":{"type":"all"},"user_input":{"type":"none"}}}),
    |tool| {
        let BuiltinMessagesTool::WebFetch20250910(fetch) = tool else { panic!("expected web_fetch_20250910") };
        assert_eq!(fetch.max_content_tokens, Some(5000));
        assert_eq!(fetch.citations.unwrap().enabled, Some(true));
        let sources = fetch.url_sources.unwrap();
        let Some(ToolResultUrlSource::Only { tools, .. }) = &sources.client_tool_results else {
            panic!("expected only client tool results");
        };
        assert!(matches!(tools.as_slice(), [UrlSourceToolReference::ToolReference { name, .. }] if name == "lookup"));
        assert!(matches!(sources.server_tool_results, Some(ToolResultUrlSource::All { .. })));
        assert!(matches!(sources.user_input, Some(UserInputUrlSource::None { .. })));
        assert!(sources.extra.is_empty());
        assert!(fetch.extra.is_empty());
    }
)]
#[case::web_fetch_20260209(
    json!({"type":"web_fetch_20260209","name":"web_fetch","url_sources":{"server_tool_results":{"type":"except","tools":[{"type":"tool_reference","name":"web_search"}]}}}),
    |tool| {
        let BuiltinMessagesTool::WebFetch20260209(fetch) = tool else { panic!("expected web_fetch_20260209") };
        assert!(matches!(
            fetch.url_sources.unwrap().server_tool_results,
            Some(ToolResultUrlSource::Except { tools, .. }) if tools.len() == 1
        ));
        assert!(fetch.extra.is_empty());
    }
)]
#[case::web_fetch_20260309(
    json!({"type":"web_fetch_20260309","name":"web_fetch","use_cache":false}),
    |tool| {
        let BuiltinMessagesTool::WebFetch20260309(fetch) = tool else { panic!("expected web_fetch_20260309") };
        assert_eq!(fetch.use_cache, Some(false));
        assert!(fetch.extra.is_empty());
    }
)]
#[case::web_fetch_20260318(
    json!({"type":"web_fetch_20260318","name":"web_fetch","use_cache":true,"response_inclusion":"full"}),
    |tool| {
        let BuiltinMessagesTool::WebFetch20260318(fetch) = tool else { panic!("expected web_fetch_20260318") };
        assert_eq!(fetch.use_cache, Some(true));
        assert_eq!(fetch.response_inclusion, Some(ResponseInclusion::Full));
        assert!(fetch.extra.is_empty());
    }
)]
#[case::advisor_20260301(
    json!({"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5","max_tokens":2048,"caching":{"type":"ephemeral","ttl":"5m"}}),
    |tool| {
        let BuiltinMessagesTool::Advisor20260301(advisor) = tool else { panic!("expected advisor_20260301") };
        assert_eq!(advisor.model, "claude-opus-5-5");
        assert_eq!(advisor.max_tokens, Some(2048));
        assert_eq!(advisor.caching.unwrap().ttl.as_deref(), Some("5m"));
        assert!(advisor.cache_control.is_none());
        assert!(advisor.extra.is_empty());
    }
)]
#[case::browser_toolset_20260801(
    json!({"type":"browser_toolset_20260801","configs":{"type":{"enabled":false},"javascript_exec":{"defer_loading":true}}}),
    |tool| {
        let BuiltinMessagesTool::BrowserToolset20260801(toolset) = tool else { panic!("expected browser_toolset_20260801") };
        let configs = toolset.configs.unwrap();
        assert_eq!(configs.type_text.unwrap().enabled, Some(false));
        assert_eq!(configs.javascript_exec.unwrap().defer_loading, Some(true));
        assert!(configs.navigate.is_none());
        assert!(configs.extra.is_empty());
        assert!(toolset.extra.is_empty());
    }
)]
#[case::computer_toolset_20260801(
    json!({"type":"computer_toolset_20260801","configs":{"zoom":{"enabled":false},"cursor_position":{"enabled":true}}}),
    |tool| {
        let BuiltinMessagesTool::ComputerToolset20260801(toolset) = tool else { panic!("expected computer_toolset_20260801") };
        let configs = toolset.configs.unwrap();
        assert_eq!(configs.zoom.unwrap().enabled, Some(false));
        assert_eq!(configs.cursor_position.unwrap().enabled, Some(true));
        assert!(configs.extra.is_empty());
        assert!(toolset.extra.is_empty());
    }
)]
#[case::mcp_toolset(
    json!({"type":"mcp_toolset","mcp_server_name":"kb","default_config":{"enabled":false},
        "configs":{"search":{"enabled":true,"defer_loading":true}},
        "tools":[{"name":"search","input_schema":{"type":"object"},"description":"Search"}]}),
    |tool| {
        let BuiltinMessagesTool::McpToolset(toolset) = tool else { panic!("expected mcp_toolset") };
        assert_eq!(toolset.mcp_server_name, "kb");
        assert_eq!(toolset.default_config.unwrap().enabled, Some(false));
        let configs = toolset.configs.unwrap();
        assert_eq!(configs["search"].enabled, Some(true));
        assert_eq!(configs["search"].defer_loading, Some(true));
        let [listed] = toolset.tools.as_deref().unwrap() else { panic!("expected one pinned tool") };
        assert_eq!(listed.description.as_deref(), Some("Search"));
        assert!(toolset.extra.is_empty());
    }
)]
fn builtin_tools_decode_typed_definitions(
    #[case] wire: Value,
    #[case] check: fn(BuiltinMessagesTool),
) {
    check(round_trip::<BuiltinMessagesTool>(wire));
}

#[rstest]
fn builtin_tools_preserve_unknown_fields() {
    let tool = round_trip::<BuiltinMessagesTool>(
        json!({"type":"bash_20250124","name":"bash","extension":[1,null]}),
    );
    let BuiltinMessagesTool::Bash20250124(bash) = tool else {
        panic!("expected bash tool");
    };
    assert_eq!(bash.extra.get("extension"), Some(&json!([1, null])));
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
#[case::missing_discriminator(json!({"name":"bash"}))]
#[case::unknown_discriminator(json!({"type":"future_tool","name":"bash"}))]
#[case::null_discriminator(json!({"type":null,"name":"bash"}))]
#[case::bash_without_name(json!({"type":"bash_20250124"}))]
#[case::bash_wrong_name(json!({"type":"bash_20241022","name":"shell"}))]
#[case::legacy_editor_with_new_name(json!({"type":"text_editor_20250124","name":"str_replace_based_edit_tool"}))]
#[case::new_editor_with_legacy_name(json!({"type":"text_editor_20250728","name":"str_replace_editor"}))]
#[case::editor_20250429_with_legacy_name(json!({"type":"text_editor_20250429","name":"str_replace_editor"}))]
#[case::memory_without_name(json!({"type":"memory_20250818"}))]
#[case::computer_without_width(json!({"type":"computer_20250124","name":"computer","display_height_px":768}))]
#[case::computer_without_height(json!({"type":"computer_20241022","name":"computer","display_width_px":1024}))]
#[case::zoom_computer_without_display(json!({"type":"computer_20251124","name":"computer"}))]
#[case::code_execution_without_name(json!({"type":"code_execution_20250825"}))]
#[case::regex_search_with_bm25_name(json!({"type":"tool_search_tool_regex","name":"tool_search_tool_bm25"}))]
#[case::bm25_search_with_regex_name(json!({"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_regex"}))]
#[case::web_search_with_fetch_name(json!({"type":"web_search_20250305","name":"web_fetch"}))]
#[case::web_fetch_without_name(json!({"type":"web_fetch_20260318"}))]
#[case::advisor_without_model(json!({"type":"advisor_20260301","name":"advisor"}))]
#[case::advisor_without_name(json!({"type":"advisor_20260301","model":"claude-opus-5-5"}))]
#[case::mcp_toolset_without_server(json!({"type":"mcp_toolset"}))]
#[case::unknown_allowed_caller(json!({"type":"code_execution_20250825","name":"code_execution","allowed_callers":["code_execution_20250522"]}))]
#[case::unknown_response_inclusion(json!({"type":"web_search_20260318","name":"web_search","response_inclusion":"partial"}))]
#[case::user_input_only_filter(json!({"type":"web_fetch_20250910","name":"web_fetch","url_sources":{"user_input":{"type":"only","tools":[]}}}))]
#[case::negative_limit(json!({"type":"web_search_20250305","name":"web_search","max_uses":-1}))]
#[case::wrong_mcp_config(json!({"type":"mcp_toolset","mcp_server_name":"kb","configs":{"search":{"enabled":"yes"}}}))]
#[case::wrong_browser_config(json!({"type":"browser_toolset_20260801","configs":{"navigate":{"enabled":1}}}))]
fn builtin_tools_reject_malformed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<BuiltinMessagesTool>(wire.clone()).is_err());
    assert!(serde_json::from_value::<MessagesToolParam>(wire).is_err());
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
    let WebFetchDocument::Document(document) = &fetched.content;
    assert!(matches!(&document.source, ContentSource::Text { data, .. } if data == "page"));
    assert!(fetched.extra.is_empty());
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
    let MessagesContentPart::BashCodeExecutionToolResult(bash) = part(json!({
        "type":"bash_code_execution_tool_result",
        "tool_use_id":"srvtoolu_7",
        "content":{"type":"bash_code_execution_result","stdout":"ok","stderr":"","return_code":0,
            "content":[{"type":"bash_code_execution_output","file_id":"file_2"}]}
    })) else {
        panic!("expected bash code execution result");
    };
    let BashCodeExecutionToolResultContent::BashCodeExecutionResult(ran) = &bash.content else {
        panic!("expected bash execution result");
    };
    assert_eq!((ran.stdout.as_str(), ran.return_code), ("ok", 0));
    assert!(
        matches!(ran.content.as_slice(), [BashCodeExecutionOutput::BashCodeExecutionOutput { file_id, .. }] if file_id == "file_2")
    );
    assert!(ran.extra.is_empty());
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
    let MessagesContentPart::ToolSearchToolResult(found) = part(json!({
        "type":"tool_search_tool_result",
        "tool_use_id":"srvtoolu_8",
        "content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"lookup"}]}
    })) else {
        panic!("expected tool search result");
    };
    let ToolSearchToolResultContent::ToolSearchToolSearchResult(result) = &found.content else {
        panic!("expected tool search hits");
    };
    let [ToolSearchReference::ToolReference(reference)] = result.tool_references.as_slice() else {
        panic!("expected one tool reference");
    };
    assert_eq!(reference.tool_name, "lookup");
    assert!(reference.extra.is_empty() && result.extra.is_empty());
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
    assert_eq!(
        mcp_result.content,
        Some(McpToolResultContent::Text("failed".into()))
    );
    let MessagesContentPart::McpToolResult(text_result) = part(json!({
        "type":"mcp_tool_result",
        "tool_use_id":"mcptoolu_3",
        "content":[{"type":"text","text":"found"}]
    })) else {
        panic!("expected MCP tool result");
    };
    let Some(McpToolResultContent::Blocks(blocks)) = &text_result.content else {
        panic!("expected MCP text blocks");
    };
    let [McpToolResultText::Text(text)] = blocks.as_slice() else {
        panic!("expected one text block");
    };
    assert_eq!(text.text, "found");
    let MessagesContentPart::McpToolResult(empty_result) =
        part(json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_2"}))
    else {
        panic!("expected MCP tool result");
    };
    assert!(empty_result.content.is_none());
    assert!(empty_result.extra.is_empty());
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
    assert_eq!(fallback.from.model, "claude-opus-5-5");
    assert_eq!(fallback.to.model, "claude-sonnet-5-5");
    let Some(FallbackTrigger::Refusal { category, extra }) = &fallback.trigger else {
        panic!("expected refusal trigger");
    };
    assert_eq!(category.as_deref(), Some("cyber"));
    assert!(extra.is_empty() && fallback.extra.is_empty());
    let MessagesContentPart::Fallback(untriggered) = part(json!({
        "type":"fallback",
        "from":{"model":"claude-opus-5-5"},
        "to":{"model":"claude-sonnet-5-5"}
    })) else {
        panic!("expected fallback block");
    };
    assert!(untriggered.trigger.is_none());
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
#[case::tool_search_non_reference(json!({"type":"tool_search_tool_result","tool_use_id":"t","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"text","text":"lookup"}]}}))]
#[case::web_fetch_text_content(json!({"type":"web_fetch_tool_result","tool_use_id":"t","content":{"type":"web_fetch_result","url":"https://example.test","content":{"type":"text","text":"page"}}}))]
#[case::bash_result_with_code_output(json!({"type":"bash_code_execution_tool_result","tool_use_id":"t","content":{"type":"bash_code_execution_result","stdout":"","stderr":"","return_code":0,"content":[{"type":"code_execution_output","file_id":"f"}]}}))]
#[case::code_result_with_bash_output(json!({"type":"code_execution_tool_result","tool_use_id":"t","content":{"type":"code_execution_result","stdout":"","stderr":"","return_code":0,"content":[{"type":"bash_code_execution_output","file_id":"f"}]}}))]
#[case::fallback_unknown_trigger(json!({"type":"fallback","from":{"model":"a"},"to":{"model":"b"},"trigger":{"type":"overload"}}))]
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
        "allowed_callers":["direct","code_execution_20260120"],
        "extension":{"nested":[1,null]}
    }));
    assert_eq!(
        tool.allowed_callers.as_deref(),
        Some([AllowedCaller::Direct, AllowedCaller::CodeExecution20260120].as_slice())
    );
    assert_eq!(tool.tool_type, Some(CustomToolType::Custom));
    assert_eq!(tool.name, "lookup");
    assert_eq!((tool.strict, tool.defer_loading), (Some(false), Some(true)));
    let JsonSchema::Object(schema) = &tool.input_schema else {
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
#[case::unknown_allowed_caller(json!({"name":"lookup","input_schema":{},"allowed_callers":["anyone"]}))]
fn custom_tool_rejects_invalid_shapes(#[case] wire: Value) {
    assert!(serde_json::from_value::<CustomTool>(wire).is_err());
}

#[rstest]
#[case::builtin(json!({"type":"web_search_20250305","name":"web_search","max_uses":2}), true)]
#[case::builtin_toolset(json!({"type":"mcp_toolset","mcp_server_name":"kb"}), true)]
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
#[case::image(json!([{"type":"image","source":{"type":"url","url":"https://example.test/a.png"}}]))]
#[case::tool_reference(json!([{"type":"tool_reference","tool_name":"lookup"}]))]
#[case::untagged(json!([{"text":"found"}]))]
fn mcp_tool_result_content_rejects_non_text_blocks(#[case] content: Value) {
    assert!(
        serde_json::from_value::<MessagesContentPart>(
            json!({"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","content":content})
        )
        .is_err()
    );
}

#[rstest]
#[case::model(json!({"type":"model_changed","cache_missed_input_tokens":12}), Some(12))]
#[case::system(json!({"type":"system_changed","cache_missed_input_tokens":3}), Some(3))]
#[case::tools(json!({"type":"tools_changed","cache_missed_input_tokens":0}), Some(0))]
#[case::messages(json!({"type":"messages_changed","cache_missed_input_tokens":9}), Some(9))]
#[case::not_found(json!({"type":"previous_message_not_found"}), None)]
#[case::unavailable(json!({"type":"unavailable"}), None)]
fn diagnostics_expose_cache_miss_reason(#[case] reason: Value, #[case] missed: Option<u64>) {
    let diagnostics = round_trip::<MessagesDiagnostics>(json!({"cache_miss_reason":reason}));
    let Some(Recognized::Known(reason)) = &diagnostics.cache_miss_reason else {
        panic!("expected known cache miss reason");
    };
    let actual = match reason {
        CacheMissReason::ModelChanged(tokens)
        | CacheMissReason::SystemChanged(tokens)
        | CacheMissReason::ToolsChanged(tokens)
        | CacheMissReason::MessagesChanged(tokens) => {
            assert!(tokens.extra.is_empty());
            Some(tokens.cache_missed_input_tokens)
        }
        CacheMissReason::PreviousMessageNotFound { extra }
        | CacheMissReason::Unavailable { extra } => {
            assert!(extra.is_empty());
            None
        }
    };
    assert_eq!(actual, missed);
    assert!(diagnostics.extra.is_empty());
}

#[rstest]
#[case::model(json!({"type":"model_changed","cache_missed_input_tokens":12}), "model_changed")]
#[case::system(json!({"type":"system_changed","cache_missed_input_tokens":12}), "system_changed")]
#[case::tools(json!({"type":"tools_changed","cache_missed_input_tokens":12}), "tools_changed")]
#[case::messages(json!({"type":"messages_changed","cache_missed_input_tokens":12}), "messages_changed")]
fn cache_miss_reason_variants_keep_their_tags(#[case] reason: Value, #[case] tag: &str) {
    let parsed: CacheMissReason = serde_json::from_value(reason).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap()["type"], json!(tag));
}

#[rstest]
#[case::pending(json!({"cache_miss_reason":null}))]
#[case::future_reason(json!({"cache_miss_reason":{"type":"future_changed","cache_missed_input_tokens":1}}))]
#[case::missing_tokens(json!({"cache_miss_reason":{"type":"model_changed"}}))]
fn diagnostics_keep_pending_and_unmodeled_reasons(#[case] wire: Value) {
    let diagnostics: MessagesDiagnostics = serde_json::from_value(wire.clone()).unwrap();
    match &diagnostics.cache_miss_reason {
        None => assert_eq!(serde_json::to_value(&diagnostics).unwrap(), json!({})),
        Some(Recognized::Unrecognized(kept)) => {
            assert_eq!(kept, &wire["cache_miss_reason"]);
            assert_eq!(serde_json::to_value(&diagnostics).unwrap(), wire);
        }
        Some(Recognized::Known(reason)) => panic!("unexpected known reason {reason:?}"),
    }
}

#[rstest]
#[case::previous(json!({"previous_message_id":"msg_1"}), Some(Some("msg_1")))]
#[case::first_turn(json!({"previous_message_id":null}), Some(None))]
#[case::absent(json!({}), None)]
fn diagnostics_param_distinguishes_null_from_absent_previous_message(
    #[case] wire: Value,
    #[case] expected: Option<Option<&str>>,
) {
    let param = round_trip::<MessagesDiagnosticsParam>(wire);
    assert_eq!(
        param.previous_message_id.as_ref().map(Option::as_deref),
        expected
    );
    assert!(param.extra.is_empty());
}

#[rstest]
fn diagnostics_param_rejects_non_string_previous_message() {
    assert!(
        serde_json::from_value::<MessagesDiagnosticsParam>(json!({"previous_message_id":7}))
            .is_err()
    );
}

#[rstest]
fn cache_miss_reason_rejects_non_numeric_tokens() {
    assert!(
        serde_json::from_value::<CacheMissReason>(
            json!({"type":"model_changed","cache_missed_input_tokens":"many"})
        )
        .is_err()
    );
}
