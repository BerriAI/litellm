use litellm_llms_types::formats::messages::{ContentBlock, ContentBlockType, MessageRole};
use litellm_llms_types::serde_compat::Nullable;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::user("user", MessageRole::User)]
#[case::assistant("assistant", MessageRole::Assistant)]
#[case::system("system", MessageRole::System)]
#[case::unknown("future_role", MessageRole::Other("future_role".into()))]
#[case::case_sensitive("Assistant", MessageRole::Other("Assistant".into()))]
#[case::empty("", MessageRole::Other(String::new()))]
fn message_role_round_trips_as_a_string(#[case] wire: &str, #[case] expected: MessageRole) {
    let parsed: MessageRole = serde_json::from_value(json!(wire)).unwrap();
    assert_eq!(parsed, expected);
    assert_eq!(serde_json::to_value(parsed).unwrap(), json!(wire));
}

#[rstest]
#[case::null(json!(null))]
#[case::number(json!(1))]
#[case::boolean(json!(true))]
#[case::array(json!(["user"]))]
#[case::object(json!({"role": "user"}))]
fn message_role_rejects_non_string_json(#[case] value: Value) {
    assert!(serde_json::from_value::<MessageRole>(value).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn message_role_schema_remains_string_compatible() {
    let schema = schemars::schema_for!(MessageRole).to_value();
    assert_eq!(schema.get("title"), Some(&json!(stringify!(MessageRole))));
    assert_eq!(schema.get("type"), Some(&json!("string")));
}

#[rstest]
#[case::null(json!(null))]
#[case::number(json!(1))]
#[case::boolean(json!(true))]
#[case::array(json!(["tool_use"]))]
#[case::object(json!({"type": "tool_use"}))]
fn content_block_type_rejects_non_string_json(#[case] value: Value) {
    assert!(serde_json::from_value::<ContentBlockType>(value).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn content_block_type_schema_remains_a_string() {
    let schema = schemars::schema_for!(ContentBlockType).to_value();
    assert_eq!(schema.get("type"), Some(&json!("string")));
    assert_eq!(
        schema.get("title"),
        Some(&json!(stringify!(ContentBlockType)))
    );
}

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
    assert_eq!(
        block.block_type.as_ref().and_then(Nullable::value),
        Some(&expected)
    );
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
#[case::image(json!({"type":"image","source":{"type":"url","url":"https://example.test/image","future":null}}))]
#[case::document(json!({"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"title":"title","context":"context","citations":{"enabled":true}}))]
#[case::nested_document(json!({"type":"document","source":{"type":"content","content":[{"type":"text","text":"doc"},{"type":"image","source":{"type":"file","file_id":"file_1"}}]}}))]
#[case::container_upload(json!({"type":"container_upload","file_id":"file_1"}))]
#[case::tool_result(json!({"type":"tool_result","tool_use_id":"tool_1","is_error":true,"content":[{"type":"tool_reference","tool_name":"lookup"}]}))]
#[case::programmatic_call(json!({"type":"tool_use","id":"tool_1","caller":{"type":"code_execution_20250825","tool_id":"server_1"},"input":{"arbitrary":[1,null]}}))]
#[case::search_result(json!({"type":"web_search_result","url":"https://example.test","title":"result","encrypted_content":"opaque","page_age":null}))]
#[case::search_error(json!({"type":"web_search_tool_result","tool_use_id":"server_1","content":{"type":"web_search_tool_result_error","error_code":"unavailable","future":null}}))]
#[case::nullable_compaction(json!({"type":"compaction","content":null,"signature":"signed"}))]
#[case::future_source(json!({"type":"image","source":{"type":"future","payload":[1,null]}}))]
#[case::wrong_source_shape(json!({"type":"image","source":17,"citations":"future"}))]
fn nested_content_and_extensions_round_trip(#[case] wire: Value) {
    let block: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(block).unwrap(), wire);
}

#[rstest]
fn nested_content_is_available_without_reparsing_json() {
    use litellm_llms_types::{
        formats::messages::{BlockContent, ContentSource, ToolCaller},
        recognized::Recognized,
    };

    let block: ContentBlock = serde_json::from_value(json!({
        "type":"tool_result", "is_error":false,
        "content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}],
        "caller":{"type":"code_execution_20250825","tool_id":"server_1"}
    })).unwrap();
    assert_eq!(block.is_error, Some(Recognized::Known(false)));
    assert!(
        matches!(block.caller.as_ref().and_then(Recognized::known), Some(ToolCaller::CodeExecution { tool_id, .. }) if tool_id == "server_1")
    );
    let Some(Recognized::Known(BlockContent::Blocks(children))) = &block.content else {
        panic!("expected nested blocks");
    };
    let image = children[0].known().unwrap();
    assert_eq!(
        image.block_type,
        Some(Nullable::Value(ContentBlockType::Image))
    );
    assert!(
        matches!(image.source.as_ref().and_then(Recognized::known), Some(ContentSource::Base64 { media_type, data, .. }) if media_type == "image/png" && data == "AA==")
    );
}

#[rstest]
#[case::typed(json!({
    "max_tokens":1024,"system":"be concise",
    "metadata":{"user_id":"user"},
    "stop_sequences":["STOP"],"stream":true,"temperature":0.4,"top_p":0.9,"top_k":40,
    "tools":[{"name":"lookup","input_schema":{"type":"object"}}],
    "tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true},
    "mcp_servers":[{"type":"url","name":"server","url":"https://example.test","tool_configuration":{"allowed_tools":["lookup"]}}],
    "thinking":{"type":"enabled","budget_tokens":1024},
    "service_tier":"auto",
    "container":{"id":"container_1","expires_at":null,"skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]},
    "context_management":{"edits":[]},
    "output_format":{"type":"json_schema","schema":{"type":"object"},"strict":true},
    "speed":"fast",
    "inference_geo":"us",
    "reasoning_effort":"high",
    "compaction":{"type":"summarize","instructions":"summarize"},
    "output_config":{"format":{"type":"json_schema","schema":{"type":"object"},"strict":true}},
    "cache_control":{"type":"ephemeral"},
    "safeguards":[{"type":"dangerous_tool_use","classifier_context":{"future":null}}]
}))]
#[case::unknown_and_wrong_shapes(json!({
    "metadata":false,"tool_choice":{"type":"future"},"thinking":false,"mcp_servers":[false],
    "container":"container_id","context_management":false,"compaction":17,"output_format":[],
    "output_config":{"format":"future"},"speed":"turbo","reasoning_effort":17,
    "cache_control":false,"safeguards":"future"
}))]
#[case::missing(json!({}))]
#[case::explicit_nulls(json!({
    "max_tokens":null,"system":null,"metadata":null,"stop_sequences":null,"stream":null,
    "temperature":null,"top_p":null,"top_k":null,"tools":null,"tool_choice":null,
    "thinking":null,"service_tier":null,"container":null,"mcp_servers":null,
    "context_management":null,"output_format":null,"output_config":null,"speed":null,
    "inference_geo":null,"reasoning_effort":null,"compaction":null,
    "cache_control":null,"safeguards":null
}))]
fn optional_contracts_preserve_wire_values(#[case] params: Value) {
    use litellm_llms_types::formats::messages::MessagesOptionalParams;

    let parsed: MessagesOptionalParams = serde_json::from_value(params.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), params);
}

#[rstest]
#[case::nested_explicit_null(json!({"output_config":{"format":null}}))]
fn optional_contracts_preserve_nested_explicit_nulls(#[case] params: Value) {
    use litellm_llms_types::formats::messages::MessagesOptionalParams;

    let parsed: MessagesOptionalParams = serde_json::from_value(params.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), params);
}

#[rstest]
fn tool_choice_and_metadata_expose_typed_fields() {
    use litellm_llms_types::{
        formats::messages::{MessagesOptionalParams, ToolChoiceType},
        recognized::Recognized,
    };

    let parsed: MessagesOptionalParams = serde_json::from_value(json!({
        "metadata":{"user_id":"user"},
        "tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true},
        "cache_control":{"type":"ephemeral"}
    }))
    .unwrap();
    let choice = parsed.tool_choice.as_ref().unwrap().known().unwrap();
    assert_eq!(choice.choice_type, ToolChoiceType::Tool);
    assert_eq!(choice.name, Some(Recognized::Known("lookup".into())));
    assert_eq!(
        choice.disable_parallel_tool_use,
        Some(Recognized::Known(true))
    );
    assert_eq!(
        parsed.metadata.as_ref().unwrap().known().unwrap().user_id,
        Some(Recognized::Known("user".into()))
    );
}

#[rstest]
fn mcp_container_compaction_and_output_format_are_typed() {
    use litellm_llms_types::{
        formats::messages::{
            CompactionType, McpServerType, MessagesOptionalParams, OutputFormatType, SkillType,
        },
        recognized::Recognized,
    };

    let wire = json!({
        "mcp_servers":[{"type":"url","name":"server","url":"https://example.test","tool_configuration":{"allowed_tools":["lookup"]}}],
        "container":{"id":"container_1","skills":[{"type":"custom","skill_id":"skill_1","version":"1"}]},
        "compaction":{"type":"summarize","instructions":"summarize"},
        "output_format":{"type":"json_schema","schema":{"type":"object"},"strict":true}
    });
    let parsed: MessagesOptionalParams = serde_json::from_value(wire.clone()).unwrap();
    let server = parsed
        .mcp_servers
        .as_ref()
        .and_then(Nullable::value)
        .unwrap()[0]
        .known()
        .unwrap();
    assert_eq!(server.server_type, McpServerType::Url);
    assert_eq!(
        server.url,
        Some(Recognized::Known("https://example.test".into()))
    );
    assert_eq!(
        server
            .tool_configuration
            .as_ref()
            .unwrap()
            .known()
            .unwrap()
            .allowed_tools,
        Some(Recognized::Known(vec!["lookup".into()]))
    );
    let litellm_llms_types::formats::messages::ContainerReference::Parameters(container) =
        parsed.container.as_ref().unwrap().known().unwrap()
    else {
        panic!("expected container parameters")
    };
    let skill = container.skills.as_ref().unwrap().known().unwrap()[0]
        .known()
        .unwrap();
    assert_eq!(skill.skill_type, SkillType::Custom);
    assert_eq!(skill.skill_id, Some(Recognized::Known("skill_1".into())));
    let compaction = parsed.compaction.as_ref().unwrap().known().unwrap();
    assert_eq!(compaction.compaction_type, CompactionType::Summarize);
    assert_eq!(
        compaction.instructions,
        Some(Recognized::Known("summarize".into()))
    );
    let format = parsed.output_format.as_ref().unwrap().known().unwrap();
    assert_eq!(format.format_type, OutputFormatType::JsonSchema);
    assert_eq!(
        serde_json::to_value(&format.schema).unwrap(),
        json!({"type":"object"})
    );
    assert_eq!(format.strict, Some(Recognized::Known(true)));
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
fn custom_tools_expose_schema_and_execution_options() {
    use litellm_llms_types::{
        formats::messages::{MessagesTool, ToolDefinition},
        recognized::Recognized,
    };

    let wire = json!({
        "name":"lookup","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false,"$defs":{"query":{"type":"string"}}},
        "strict":true,"defer_loading":true,"allowed_callers":["direct"],"input_examples":[{"query":"test"}],"eager_input_streaming":true,"future":null
    });
    let tool: Recognized<MessagesTool> = serde_json::from_value(wire.clone()).unwrap();
    let Some(MessagesTool::Custom(custom)) = tool.known() else {
        panic!("expected a custom tool")
    };
    let ToolDefinition {
        input_schema,
        strict,
        defer_loading,
        allowed_callers,
        input_examples,
        eager_input_streaming,
        ..
    } = &custom.definition;
    let litellm_llms_types::json_schema::JsonSchema::Object(schema) =
        input_schema.as_ref().unwrap().known().unwrap()
    else {
        panic!("expected an object schema")
    };
    assert_eq!(
        schema.schema_type,
        Some(Recognized::Known(
            litellm_llms_types::json_schema::JsonSchemaType::Name("object".into())
        ))
    );
    assert_eq!(
        schema.required,
        Some(Recognized::Known(vec!["query".into()]))
    );
    assert!(
        matches!(&schema.additional_properties, Some(Recognized::Known(value)) if **value == litellm_llms_types::json_schema::JsonSchema::Boolean(false))
    );
    assert_eq!(
        serde_json::to_value(&schema.properties.as_ref().unwrap().known().unwrap()["query"])
            .unwrap(),
        json!({"type":"string"})
    );
    assert_eq!(
        serde_json::to_value(&schema.defs.as_ref().unwrap().known().unwrap()["query"]).unwrap(),
        json!({"type":"string"})
    );
    assert_eq!(*strict, Some(Recognized::Known(true)));
    assert_eq!(*defer_loading, Some(Recognized::Known(true)));
    assert_eq!(
        *allowed_callers,
        Some(Recognized::Known(vec!["direct".into()]))
    );
    assert_eq!(
        input_examples.as_ref().unwrap().known().unwrap()[0]["query"],
        json!("test")
    );
    assert_eq!(*eager_input_streaming, Some(Recognized::Known(true)));
    assert_eq!(serde_json::to_value(tool).unwrap(), wire);
}

#[rstest]
#[case::future_tool(json!({"type":"future_tool","name":"lookup","payload":null}))]
#[case::scalar_tool(json!(17))]
fn unknown_tools_are_kept_as_raw_values(#[case] wire: Value) {
    use litellm_llms_types::{formats::messages::MessagesTool, recognized::Recognized};

    let parsed: Recognized<MessagesTool> = serde_json::from_value(wire.clone()).unwrap();
    assert!(matches!(parsed, Recognized::Unrecognized(_)));
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::all_nullable_fields(json!({"type":null,"text":null,"thinking":null,"signature":null,"data":null,"id":null,"name":null,"tool_use_id":null,"input":null,"content":null,"provider_specific_fields":null,"cache_control":null}))]
#[case::cache_control_nulls(json!({"type":"text","text":"hi","cache_control":{"type":null,"ttl":null,"scope":null}}))]
fn nullable_block_fields_are_distinct_from_missing_fields(#[case] wire: Value) {
    let parsed: ContentBlock = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
    assert_eq!(
        serde_json::to_value(ContentBlock::default()).unwrap(),
        json!({})
    );
}

#[rstest]
#[case::web_search(
    "web_search_20250305",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::WebSearch
)]
#[case::web_search_v2(
    "web_search_20260209",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::WebSearch20260209
)]
#[case::computer(
    "computer_20250124",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::Computer
)]
#[case::computer_older(
    "computer_20241022",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::Computer20241022
)]
#[case::bash(
    "bash_20250124",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::Bash
)]
#[case::bash_older(
    "bash_20241022",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::Bash20241022
)]
#[case::text_editor(
    "text_editor_20250728",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::TextEditor
)]
#[case::text_editor_20250124(
    "text_editor_20250124",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::TextEditor20250124
)]
#[case::text_editor_20241022(
    "text_editor_20241022",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::TextEditor20241022
)]
#[case::code_execution(
    "code_execution_20250825",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::CodeExecution
)]
#[case::code_execution_older(
    "code_execution_20250522",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::CodeExecution20250522
)]
#[case::memory(
    "memory_20250818",
    litellm_llms_types::formats::messages::BuiltinMessagesTool::Memory
)]
fn builtin_tools_have_typed_options(
    #[case] tool_type: &str,
    #[case] constructor: fn(
        litellm_llms_types::formats::messages::ToolDefinition,
    ) -> litellm_llms_types::formats::messages::BuiltinMessagesTool,
) {
    use litellm_llms_types::{
        formats::messages::{MessagesTool, ToolDefinition},
        recognized::Recognized,
    };

    let wire = json!({"type":tool_type,"name":"tool","defer_loading":true,"future":null});
    let parsed: Recognized<MessagesTool> = serde_json::from_value(wire.clone()).unwrap();
    let expected = ToolDefinition {
        name: Some(Recognized::Known("tool".into())),
        defer_loading: Some(Recognized::Known(true)),
        extra: serde_json::Map::from_iter([("future".into(), Value::Null)]),
        ..ToolDefinition::default()
    };
    assert_eq!(
        parsed,
        Recognized::Known(MessagesTool::Builtin(constructor(expected)))
    );
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::missing(json!({"type":"compact_20260112"}))]
#[case::input_tokens(json!({"type":"compact_20260112","trigger":{"type":"input_tokens","value":1000,"future":null},"instructions":"keep code"}))]
#[case::negative_tokens(json!({"type":"compact_20260112","trigger":{"type":"input_tokens","value":-1000}}))]
#[case::null_trigger(json!({"type":"compact_20260112","trigger":null}))]
#[case::unknown_trigger(json!({"type":"compact_20260112","trigger":{"type":"future","value":[1,null]}}))]
#[case::scalar_trigger(json!({"type":"compact_20260112","trigger":false}))]
#[case::missing_trigger_value(json!({"type":"compact_20260112","trigger":{"type":"input_tokens"}}))]
#[case::wrong_trigger_value(json!({"type":"compact_20260112","trigger":{"type":"input_tokens","value":"1000"}}))]
#[case::null_trigger_value(json!({"type":"compact_20260112","trigger":{"type":"input_tokens","value":null}}))]
fn compaction_triggers_preserve_wire_values_and_edit_recognition(#[case] wire: Value) {
    use litellm_llms_types::{formats::messages::ContextEdit, recognized::Recognized};

    let parsed: Recognized<ContextEdit> = serde_json::from_value(wire.clone()).unwrap();
    assert!(matches!(
        parsed,
        Recognized::Known(ContextEdit::Compact { .. })
    ));
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}

#[rstest]
#[case::integer(json!(1000), litellm_llms_types::recognized::Recognized::Known(1000))]
#[case::negative(json!(-1000), litellm_llms_types::recognized::Recognized::Known(-1000))]
#[case::string(json!("1000"), litellm_llms_types::recognized::Recognized::Unrecognized(json!("1000")))]
#[case::null(json!(null), litellm_llms_types::recognized::Recognized::Unrecognized(json!(null)))]
fn compaction_exposes_typed_trigger_values(
    #[case] value: Value,
    #[case] expected: litellm_llms_types::recognized::Recognized<i64>,
) {
    use litellm_llms_types::{
        formats::messages::{ContextEdit, ContextTrigger},
        recognized::Recognized,
    };

    let wire = json!({"type":"compact_20260112","trigger":{"type":"input_tokens","value":value,"future":null}});
    let parsed: ContextEdit = serde_json::from_value(wire.clone()).unwrap();
    let ContextEdit::Compact {
        trigger: Some(Recognized::Known(ContextTrigger::InputTokens { value, extra })),
        ..
    } = &parsed
    else {
        panic!("expected a typed input-token trigger")
    };
    assert_eq!(value, &expected);
    assert_eq!(extra.get("future"), Some(&Value::Null));
    assert_eq!(serde_json::to_value(parsed).unwrap(), wire);
}
