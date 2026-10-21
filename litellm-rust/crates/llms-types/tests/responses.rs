use litellm_llms_types::formats::chat_completions::{PromptCacheBreakpoint, PromptCacheMode};
use litellm_llms_types::formats::responses::{
    ResponsesAdditionalTools, ResponsesApplyPatchCall, ResponsesApplyPatchCallOutput,
    ResponsesApplyPatchOperation, ResponsesCodeOutput, ResponsesCompaction,
    ResponsesComputerAction, ResponsesComputerCall, ResponsesComputerCallOutput,
    ResponsesComputerOutput, ResponsesContentPart, ResponsesCoordinate, ResponsesCustomToolCall,
    ResponsesCustomToolCallOutput, ResponsesFunctionCall, ResponsesFunctionCallOutput,
    ResponsesImageGenerationCall, ResponsesInputContent, ResponsesLocalShellAction,
    ResponsesLocalShellCall, ResponsesLocalShellCallOutput, ResponsesMcpApprovalRequest,
    ResponsesMcpApprovalResponse, ResponsesMcpError, ResponsesMcpErrorDetail, ResponsesOutputItem,
    ResponsesProgram, ResponsesProgramOutput, ResponsesSafetyCheck, ResponsesShellAction,
    ResponsesShellCall, ResponsesShellCallOutput, ResponsesShellEnvironment, ResponsesShellOutcome,
    ResponsesShellOutputChunk, ResponsesToolCaller, ResponsesToolOutput, ResponsesToolSearchCall,
    ResponsesToolSearchOutput, ResponsesWebSearchAction,
    streaming_websocket::{ResponsesEventResponse, ResponsesWsEventType},
};
use litellm_llms_types::recognized::Recognized;
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
#[case::create("response.create", ResponsesWsEventType::ResponseCreate)]
#[case::created("response.created", ResponsesWsEventType::ResponseCreated)]
#[case::completed("response.completed", ResponsesWsEventType::ResponseCompleted)]
#[case::failed("response.failed", ResponsesWsEventType::ResponseFailed)]
#[case::incomplete("response.incomplete", ResponsesWsEventType::ResponseIncomplete)]
#[case::error("error", ResponsesWsEventType::Error)]
#[case::unknown(
    "response.output_text.delta",
    ResponsesWsEventType::Other("response.output_text.delta".to_string())
)]
#[case::empty("", ResponsesWsEventType::Other(String::new()))]
#[case::case_sensitive(
    "Response.Completed",
    ResponsesWsEventType::Other("Response.Completed".into())
)]
#[case::escaped("future\"\\\n", ResponsesWsEventType::Other("future\"\\\n".into()))]
fn websocket_event_type_round_trips(#[case] wire: &str, #[case] expected: ResponsesWsEventType) {
    let serialized = serde_json::to_string(&expected).unwrap();
    assert_eq!(serialized, serde_json::to_string(wire).unwrap());
    assert_eq!(
        serde_json::from_str::<ResponsesWsEventType>(&serialized).unwrap(),
        expected
    );
}

#[rstest]
#[case::number("17")]
#[case::boolean("true")]
#[case::null("null")]
#[case::array("[]")]
#[case::object("{}")]
fn websocket_event_type_rejects_non_strings(#[case] wire: &str) {
    assert!(serde_json::from_str::<ResponsesWsEventType>(wire).is_err());
}

#[cfg(feature = "schema")]
#[rstest]
fn websocket_event_type_schema_is_open_string() {
    let schema = schemars::schema_for!(ResponsesWsEventType);
    assert_eq!(
        schema.to_value().get("type"),
        Some(&serde_json::json!("string"))
    );
}

#[rstest]
#[case::message(json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.test","start_index":0,"end_index":6}]}]}))]
#[case::function_call(json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"q\"}"}))]
#[case::custom_tool(json!({"type":"custom_tool_call","name":"lookup","input":"q"}))]
#[case::reasoning(json!({"type":"reasoning","summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"opaque"}))]
#[case::web_search(json!({"type":"web_search_call","action":{"type":"search","queries":["q"],"sources":[{"type":"url","url":"https://example.test"}]}}))]
#[case::file_search(json!({"type":"file_search_call","queries":["q"],"results":[{"file_id":"file_1","score":1,"attributes":{"custom":[1,null]}}]}))]
#[case::code(json!({"type":"code_interpreter_call","outputs":[{"type":"logs","logs":"done"},{"type":"image","url":"https://example.test"}]}))]
#[case::image(json!({"type":"image_generation_call","result":"generated"}))]
#[case::mcp(json!({"type":"mcp_call","server_label":"server","name":"lookup","arguments":"{}","output":"done"}))]
fn output_items_round_trip(#[case] wire: Value) {
    let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
    match &item {
        ResponsesOutputItem::Message(message) => {
            assert_eq!(message.role.as_deref(), Some("assistant"));
            let Some(content) = &message.content else {
                panic!("expected content")
            };
            let [
                ResponsesContentPart::OutputText {
                    text,
                    annotations: Some(annotations),
                    ..
                },
            ] = content.as_slice()
            else {
                panic!("expected output text and annotations")
            };
            assert_eq!(text, "answer");
            let [
                litellm_llms_types::formats::responses::ResponsesAnnotation::UrlCitation(citation),
            ] = annotations.as_slice()
            else {
                panic!("expected URL citation")
            };
            assert_eq!(citation.url.as_deref(), Some("https://example.test"));
            assert_eq!(citation.start_index, Some(0));
        }
        ResponsesOutputItem::FunctionCall(call) => {
            assert_eq!(call.call_id.as_deref(), Some("call_1"));
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.arguments.as_deref(), Some("{\"query\":\"q\"}"));
        }
        ResponsesOutputItem::CustomToolCall(call) => {
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.input.as_deref(), Some("q"));
        }
        ResponsesOutputItem::Reasoning(reasoning) => {
            assert_eq!(reasoning.encrypted_content.as_deref(), Some("opaque"));
            let Some(summary) = &reasoning.summary else {
                panic!("expected summary")
            };
            let [ResponsesContentPart::SummaryText { text, .. }] = summary.as_slice() else {
                panic!("expected summary text")
            };
            assert_eq!(text, "summary");
        }
        ResponsesOutputItem::WebSearchCall(call) => {
            let Some(ResponsesWebSearchAction::Search {
                queries: Some(queries),
                sources: Some(sources),
                ..
            }) = &call.action
            else {
                panic!("expected search action")
            };
            assert_eq!(queries, &["q"]);
            assert_eq!(sources[0].url.as_deref(), Some("https://example.test"));
        }
        ResponsesOutputItem::FileSearchCall(call) => {
            assert_eq!(call.queries.as_deref(), Some(["q".to_owned()].as_slice()));
            let Some(results) = &call.results else {
                panic!("expected search results")
            };
            assert_eq!(results[0].file_id.as_deref(), Some("file_1"));
            assert_eq!(results[0].score, Some(1.into()));
            assert_eq!(
                results[0].attributes.as_ref().unwrap()["custom"],
                json!([1, null])
            );
        }
        ResponsesOutputItem::CodeInterpreterCall(call) => {
            let Some(outputs) = &call.outputs else {
                panic!("expected code outputs")
            };
            let [
                ResponsesCodeOutput::Logs { logs, .. },
                ResponsesCodeOutput::Image { url, .. },
            ] = outputs.as_slice()
            else {
                panic!("expected logs and image")
            };
            assert_eq!(logs, "done");
            assert_eq!(url, "https://example.test");
        }
        ResponsesOutputItem::ImageGenerationCall(call) => {
            assert_eq!(call.result.as_deref(), Some("generated"))
        }
        ResponsesOutputItem::McpCall(call) => {
            assert_eq!(call.server_label.as_deref(), Some("server"));
            assert_eq!(call.name.as_deref(), Some("lookup"));
            assert_eq!(call.arguments.as_deref(), Some("{}"));
            assert_eq!(call.output.as_deref(), Some("done"));
        }
        other => panic!("unexpected variant {other:?}"),
    }
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
}

#[rstest]
fn nested_event_response_exposes_typed_output_and_preserves_extensions() {
    let wire = json!({
        "id":"response_1",
        "model":"example-model",
        "status":"completed",
        "output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","extension":true}],
        "extension":{"nested":[1,null]}
    });
    let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(response.id.as_deref(), Some("response_1"));
    assert_eq!(response.model.as_deref(), Some("example-model"));
    let Some(output) = &response.output else {
        panic!("expected typed output");
    };
    let [Recognized::Known(ResponsesOutputItem::FunctionCall(call))] = output.as_slice() else {
        panic!("expected function call");
    };
    assert_eq!(call.name.as_deref(), Some("lookup"));
    assert_eq!(call.arguments.as_deref(), Some("{}"));
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::empty(json!({}))]
#[case::partial(json!({"id":"response_1","output":[]}))]
fn nested_event_response_accepts_partial_metadata(#[case] wire: Value) {
    round_trip::<ResponsesEventResponse>(wire);
}

#[rstest]
#[case::wrong_model(json!({"model":7}))]
#[case::wrong_status(json!({"status":false}))]
#[case::wrong_output(json!({"output":{}}))]
fn event_response_rejects_malformed_typed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesEventResponse>(wire).is_err());
}

#[rstest]
#[case::unknown_type(json!({"type":"future_item","id":"item_1","payload":[1,null]}))]
#[case::missing_tag(json!({"id":"item_1"}))]
#[case::malformed_known(json!({"type":"message","content":[{"type":"output_text","text":7}]}))]
fn event_response_keeps_unmodeled_output_items_beside_typed_ones(#[case] item: Value) {
    let wire = json!({"output":[{"type":"function_call","name":"lookup"}, item.clone()]});
    let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
    let Some(
        [
            Recognized::Known(ResponsesOutputItem::FunctionCall(call)),
            Recognized::Unrecognized(kept),
        ],
    ) = response.output.as_deref()
    else {
        panic!("expected one typed item and one preserved item");
    };
    assert_eq!(call.name.as_deref(), Some("lookup"));
    assert_eq!(kept, &item);
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::find_in_page(
    json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"}),
    json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"})
)]
#[case::legacy_find(
    json!({"type":"find","url":"https://example.test","pattern":"needle"}),
    json!({"type":"find_in_page","url":"https://example.test","pattern":"needle"})
)]
fn web_search_find_action_exposes_url_and_pattern(#[case] wire: Value, #[case] serialized: Value) {
    let action: ResponsesWebSearchAction = serde_json::from_value(wire).unwrap();
    let ResponsesWebSearchAction::FindInPage {
        url,
        pattern,
        extra,
    } = &action
    else {
        panic!("expected find_in_page action");
    };
    assert_eq!(url, "https://example.test");
    assert_eq!(pattern, "needle");
    assert!(extra.is_empty());
    assert_eq!(serde_json::to_value(action).unwrap(), serialized);
}

#[rstest]
#[case::with_url(json!({"type":"open_page","url":"https://example.test"}), Some("https://example.test"))]
#[case::without_url(json!({"type":"open_page"}), None)]
fn web_search_open_page_url_is_optional(#[case] wire: Value, #[case] expected: Option<&str>) {
    let action: ResponsesWebSearchAction = serde_json::from_value(wire.clone()).unwrap();
    let ResponsesWebSearchAction::OpenPage { url, .. } = &action else {
        panic!("expected open_page action");
    };
    assert_eq!(url.as_deref(), expected);
    assert_eq!(serde_json::to_value(action).unwrap(), wire);
}

#[rstest]
#[case::find_pattern(json!({"type":"find_in_page","url":"u","pattern":false}))]
#[case::find_missing_url(json!({"type":"find_in_page","pattern":"p"}))]
#[case::open_page_url(json!({"type":"open_page","url":7}))]
fn web_search_action_rejects_malformed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesWebSearchAction>(wire).is_err());
}

#[rstest]
fn event_response_optional_fields_omit_missing_and_null() {
    let response: ResponsesEventResponse = serde_json::from_value(json!({
        "id":null,"model":null,"status":null,"output":null,"future":null
    }))
    .unwrap();
    assert!(response.id.is_none());
    assert!(response.model.is_none());
    assert!(response.status.is_none());
    assert!(response.output.is_none());
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({"future":null})
    );
}

#[rstest]
#[case::refusal(json!({"type":"refusal","refusal":"refused","future":null}), ResponsesContentPart::Refusal { refusal:"refused".into(), extra:serde_json::Map::from_iter([("future".into(), Value::Null)]) })]
#[case::reasoning(json!({"type":"reasoning_text","text":"reason"}), ResponsesContentPart::ReasoningText { text:"reason".into(), extra:Default::default() })]
fn content_parts_expose_typed_variants(
    #[case] wire: Value,
    #[case] expected: ResponsesContentPart,
) {
    let content: ResponsesContentPart = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(content, expected);
    assert_eq!(serde_json::to_value(content).unwrap(), wire);
}

#[rstest]
fn mcp_discovery_exposes_tools_and_nested_schemas() {
    let wire = json!({
        "type":"mcp_list_tools","id":"item_1","server_label":"tools",
        "tools":[{"name":"lookup","description":"Look up a value","input_schema":{"type":"object","properties":{"query":{"type":"string"}}},"annotations":{"read_only":false},"future":null}],
        "extension":[1,null]
    });
    let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
    let ResponsesOutputItem::McpListTools(discovery) = &item else {
        panic!("expected discovery")
    };
    assert_eq!(discovery.server_label.as_deref(), Some("tools"));
    let tool = &discovery.tools.as_ref().unwrap()[0];
    assert_eq!(tool.name.as_deref(), Some("lookup"));
    let Some(litellm_llms_types::json_schema::JsonSchema::Object(schema)) = &tool.input_schema
    else {
        panic!("expected tool schema")
    };
    assert!(schema.properties.as_ref().unwrap().contains_key("query"));
    assert_eq!(tool.annotations, Some(json!({"read_only":false})));
    assert_eq!(tool.extra["future"], Value::Null);
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
}

#[rstest]
#[case::partial(json!({"server_label":"tools","tools":[]}))]
#[case::failure(json!({"server_label":"tools","error":"unavailable"}))]
fn mcp_discovery_accepts_partial_payloads(#[case] wire: Value) {
    let discovery: litellm_llms_types::formats::responses::ResponsesMcpListTools =
        serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(discovery.server_label.as_deref(), Some("tools"));
    assert!(discovery.id.is_none());
    assert_eq!(serde_json::to_value(discovery).unwrap(), wire);
}

#[rstest]
#[case::message(json!("unavailable"), ResponsesMcpError::Message("unavailable".into()))]
#[case::protocol(json!({"type":"mcp_protocol_error","code":-32600,"message":"invalid","extension":null}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpProtocolError {code:-32600,message:"invalid".into(),extra:serde_json::Map::from_iter([("extension".into(), Value::Null)])}))]
#[case::http(json!({"type":"http_error","code":503,"message":"unavailable"}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::HttpError {code:503,message:"unavailable".into(),extra:Default::default()}))]
#[case::tool(json!({"type":"mcp_tool_execution_error","content":{"arbitrary":[1,null]}}), ResponsesMcpError::Detail(ResponsesMcpErrorDetail::McpToolExecutionError {content:json!({"arbitrary":[1,null]}),extra:Default::default()}))]
fn mcp_call_errors_expose_typed_variants(#[case] wire: Value, #[case] expected: ResponsesMcpError) {
    let error: ResponsesMcpError = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(error, expected);
    assert_eq!(serde_json::to_value(&error).unwrap(), wire);
    let call: ResponsesOutputItem =
        serde_json::from_value(json!({"type":"mcp_call","error":wire})).unwrap();
    let ResponsesOutputItem::McpCall(call) = call else {
        panic!("expected call")
    };
    assert_eq!(call.error, Some(error));
}

#[rstest]
#[case::tools_not_array(json!({"type":"mcp_list_tools","tools":{}}))]
#[case::invalid_name(json!({"type":"mcp_list_tools","tools":[{"name":7}]}))]
#[case::invalid_schema(json!({"type":"mcp_list_tools","tools":[{"input_schema":{"properties":{"query":7}}}]}))]
#[case::invalid_error_code(json!({"type":"mcp_call","error":{"type":"http_error","code":"503","message":"unavailable"}}))]
#[case::missing_error_content(json!({"type":"mcp_call","error":{"type":"mcp_tool_execution_error"}}))]
#[case::unknown_error_tag(json!({"type":"mcp_call","error":{"type":"future"}}))]
fn mcp_output_rejects_malformed_known_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesOutputItem>(wire).is_err());
}

fn text(value: &str) -> Option<String> {
    Some(value.to_owned())
}

fn program_caller() -> Option<ResponsesToolCaller> {
    Some(ResponsesToolCaller::Program {
        caller_id: "prog_1".into(),
        extra: Default::default(),
    })
}

fn direct_caller() -> Option<ResponsesToolCaller> {
    Some(ResponsesToolCaller::Direct {
        extra: Default::default(),
    })
}

fn safety_check() -> ResponsesSafetyCheck {
    ResponsesSafetyCheck {
        id: text("sc_1"),
        code: text("malicious_instructions"),
        message: text("check"),
        ..Default::default()
    }
}

#[rstest]
#[case::function_call(
    json!({"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"lookup","arguments":"{}","namespace":"ns","async":true,"caller":{"type":"program","caller_id":"prog_1"}}),
    ResponsesOutputItem::FunctionCall(ResponsesFunctionCall {
        id: text("fc_1"), status: text("completed"), call_id: text("call_1"), name: text("lookup"),
        arguments: text("{}"), namespace: text("ns"), r#async: Some(true), caller: program_caller(), ..Default::default()
    })
)]
#[case::custom_tool_call(
    json!({"type":"custom_tool_call","call_id":"call_1","name":"lookup","input":"q","namespace":"ns","async":false,"caller":{"type":"direct"}}),
    ResponsesOutputItem::CustomToolCall(ResponsesCustomToolCall {
        call_id: text("call_1"), name: text("lookup"), input: text("q"), namespace: text("ns"),
        r#async: Some(false), caller: direct_caller(), ..Default::default()
    })
)]
#[case::image_generation_call(
    json!({"type":"image_generation_call","id":"ig_1","status":"completed","result":"b64","action":"edit","background":"opaque","output_format":"webp","quality":"high","revised_prompt":"a cat","size":"1536x864"}),
    ResponsesOutputItem::ImageGenerationCall(ResponsesImageGenerationCall {
        id: text("ig_1"), status: text("completed"), result: text("b64"), action: text("edit"), background: text("opaque"),
        output_format: text("webp"), quality: text("high"), revised_prompt: text("a cat"), size: text("1536x864"), ..Default::default()
    })
)]
#[case::function_call_output_text(
    json!({"type":"function_call_output","id":"fco_1","status":"completed","call_id":"call_1","output":"done","caller":{"type":"direct"},"created_by":"user_1","name":"lookup","namespace":"ns"}),
    ResponsesOutputItem::FunctionCallOutput(ResponsesFunctionCallOutput {
        id: text("fco_1"), status: text("completed"), call_id: text("call_1"), output: Some(ResponsesToolOutput::Text("done".into())),
        caller: direct_caller(), created_by: text("user_1"), name: text("lookup"), namespace: text("ns"), ..Default::default()
    })
)]
#[case::function_call_output_content(
    json!({"type":"function_call_output","call_id":"call_1","output":[
        {"type":"input_text","text":"t","prompt_cache_breakpoint":{"mode":"explicit"}},
        {"type":"input_image","detail":"low","file_id":"file_1","image_url":"https://example.test/i.png"},
        {"type":"input_file","detail":"high","file_data":"data","file_id":"file_2","file_url":"https://example.test/f","filename":"f.pdf"}
    ]}),
    ResponsesOutputItem::FunctionCallOutput(ResponsesFunctionCallOutput {
        call_id: text("call_1"),
        output: Some(ResponsesToolOutput::Content(vec![
            ResponsesInputContent::InputText {
                text: "t".into(),
                prompt_cache_breakpoint: Some(PromptCacheBreakpoint { mode: PromptCacheMode::Explicit, extra: Default::default() }),
                extra: Default::default(),
            },
            ResponsesInputContent::InputImage {
                detail: "low".into(), file_id: text("file_1"), image_url: text("https://example.test/i.png"),
                prompt_cache_breakpoint: None, extra: Default::default(),
            },
            ResponsesInputContent::InputFile {
                detail: text("high"), file_data: text("data"), file_id: text("file_2"), file_url: text("https://example.test/f"),
                filename: text("f.pdf"), prompt_cache_breakpoint: None, extra: Default::default(),
            },
        ])),
        ..Default::default()
    })
)]
#[case::custom_tool_call_output(
    json!({"type":"custom_tool_call_output","id":"cto_1","status":"completed","call_id":"call_1","output":[{"type":"input_text","text":"t"}],"caller":{"type":"program","caller_id":"prog_1"},"created_by":"user_1"}),
    ResponsesOutputItem::CustomToolCallOutput(ResponsesCustomToolCallOutput {
        id: text("cto_1"), status: text("completed"), call_id: text("call_1"),
        output: Some(ResponsesToolOutput::Content(vec![ResponsesInputContent::InputText {
            text: "t".into(), prompt_cache_breakpoint: None, extra: Default::default(),
        }])),
        caller: program_caller(), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::computer_call(
    json!({"type":"computer_call","id":"cu_1","status":"completed","call_id":"call_1",
        "pending_safety_checks":[{"id":"sc_1","code":"malicious_instructions","message":"check"}],
        "action":{"type":"click","button":"left","x":1,"y":2,"keys":["shift"]},
        "actions":[
            {"type":"double_click","x":3,"y":4},
            {"type":"drag","path":[{"x":5,"y":6},{"x":7,"y":8}],"keys":["ctrl"]},
            {"type":"keypress","keys":["enter"]},
            {"type":"move","x":-1,"y":9},
            {"type":"screenshot"},
            {"type":"scroll","scroll_x":0,"scroll_y":-10,"x":11,"y":12},
            {"type":"type","text":"hello"},
            {"type":"wait"}
        ]}),
    ResponsesOutputItem::ComputerCall(ResponsesComputerCall {
        id: text("cu_1"), status: text("completed"), call_id: text("call_1"), pending_safety_checks: Some(vec![safety_check()]),
        action: Some(ResponsesComputerAction::Click { button: "left".into(), x: 1, y: 2, keys: Some(vec!["shift".into()]), extra: Default::default() }),
        actions: Some(vec![
            ResponsesComputerAction::DoubleClick { x: 3, y: 4, keys: None, extra: Default::default() },
            ResponsesComputerAction::Drag {
                path: vec![
                    ResponsesCoordinate { x: 5, y: 6, extra: Default::default() },
                    ResponsesCoordinate { x: 7, y: 8, extra: Default::default() },
                ],
                keys: Some(vec!["ctrl".into()]),
                extra: Default::default(),
            },
            ResponsesComputerAction::Keypress { keys: vec!["enter".into()], extra: Default::default() },
            ResponsesComputerAction::Move { x: -1, y: 9, keys: None, extra: Default::default() },
            ResponsesComputerAction::Screenshot { extra: Default::default() },
            ResponsesComputerAction::Scroll { scroll_x: 0, scroll_y: -10, x: 11, y: 12, keys: None, extra: Default::default() },
            ResponsesComputerAction::Type { text: "hello".into(), extra: Default::default() },
            ResponsesComputerAction::Wait { extra: Default::default() },
        ]),
        ..Default::default()
    })
)]
#[case::computer_call_output(
    json!({"type":"computer_call_output","id":"cuo_1","status":"completed","call_id":"call_1","output":{"type":"computer_screenshot","file_id":"file_1","image_url":"https://example.test/s.png"},"acknowledged_safety_checks":[{"id":"sc_1","code":"malicious_instructions","message":"check"}],"created_by":"user_1"}),
    ResponsesOutputItem::ComputerCallOutput(ResponsesComputerCallOutput {
        id: text("cuo_1"), status: text("completed"), call_id: text("call_1"),
        output: Some(ResponsesComputerOutput::ComputerScreenshot { file_id: text("file_1"), image_url: text("https://example.test/s.png"), extra: Default::default() }),
        acknowledged_safety_checks: Some(vec![safety_check()]), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::program(
    json!({"type":"program","id":"prog_item","call_id":"prog_1","code":"run()","fingerprint":"fp"}),
    ResponsesOutputItem::Program(ResponsesProgram {
        id: text("prog_item"), call_id: text("prog_1"), code: text("run()"), fingerprint: text("fp"), ..Default::default()
    })
)]
#[case::program_output(
    json!({"type":"program_output","id":"po_1","status":"completed","call_id":"prog_1","result":"42"}),
    ResponsesOutputItem::ProgramOutput(ResponsesProgramOutput {
        id: text("po_1"), status: text("completed"), call_id: text("prog_1"), result: text("42"), ..Default::default()
    })
)]
#[case::tool_search_call(
    json!({"type":"tool_search_call","id":"ts_1","status":"completed","call_id":"call_1","arguments":{"query":["weather",null]},"execution":"server","created_by":"user_1"}),
    ResponsesOutputItem::ToolSearchCall(ResponsesToolSearchCall {
        id: text("ts_1"), status: text("completed"), call_id: text("call_1"), arguments: Some(json!({"query":["weather",null]})),
        execution: text("server"), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::tool_search_output(
    json!({"type":"tool_search_output","id":"tso_1","status":"completed","call_id":"call_1","execution":"client","tools":[{"type":"function","name":"lookup","parameters":null,"strict":true}],"created_by":"user_1"}),
    ResponsesOutputItem::ToolSearchOutput(ResponsesToolSearchOutput {
        id: text("tso_1"), status: text("completed"), call_id: text("call_1"), execution: text("client"),
        tools: Some(vec![serde_json::Map::from_iter([
            ("type".into(), json!("function")), ("name".into(), json!("lookup")),
            ("parameters".into(), Value::Null), ("strict".into(), json!(true)),
        ])]),
        created_by: text("user_1"), ..Default::default()
    })
)]
#[case::additional_tools(
    json!({"type":"additional_tools","id":"at_1","role":"developer","tools":[{"type":"local_shell"}]}),
    ResponsesOutputItem::AdditionalTools(ResponsesAdditionalTools {
        id: text("at_1"), role: text("developer"),
        tools: Some(vec![serde_json::Map::from_iter([("type".into(), json!("local_shell"))])]), ..Default::default()
    })
)]
#[case::compaction(
    json!({"type":"compaction","id":"cmp_1","encrypted_content":"opaque","created_by":"user_1"}),
    ResponsesOutputItem::Compaction(ResponsesCompaction {
        id: text("cmp_1"), encrypted_content: text("opaque"), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::local_shell_call(
    json!({"type":"local_shell_call","id":"ls_1","status":"completed","call_id":"call_1","action":{"type":"exec","command":["ls","-a"],"env":{"HOME":"/home/u"},"timeout_ms":1000,"user":"u","working_directory":"/tmp"}}),
    ResponsesOutputItem::LocalShellCall(ResponsesLocalShellCall {
        id: text("ls_1"), status: text("completed"), call_id: text("call_1"),
        action: Some(ResponsesLocalShellAction::Exec {
            command: vec!["ls".into(), "-a".into()], env: [("HOME".into(), "/home/u".into())].into(),
            timeout_ms: Some(1000), user: text("u"), working_directory: text("/tmp"), extra: Default::default(),
        }),
        ..Default::default()
    })
)]
#[case::local_shell_call_output(
    json!({"type":"local_shell_call_output","id":"lso_1","status":"completed","output":"{\"stdout\":\"x\"}"}),
    ResponsesOutputItem::LocalShellCallOutput(ResponsesLocalShellCallOutput {
        id: text("lso_1"), status: text("completed"), output: text("{\"stdout\":\"x\"}"), ..Default::default()
    })
)]
#[case::shell_call_local(
    json!({"type":"shell_call","id":"sh_1","status":"in_progress","call_id":"call_1","action":{"commands":["ls"],"max_output_length":100,"timeout_ms":500},"environment":{"type":"local"},"caller":{"type":"direct"},"created_by":"user_1"}),
    ResponsesOutputItem::ShellCall(ResponsesShellCall {
        id: text("sh_1"), status: text("in_progress"), call_id: text("call_1"),
        action: Some(ResponsesShellAction { commands: Some(vec!["ls".into()]), max_output_length: Some(100), timeout_ms: Some(500), ..Default::default() }),
        environment: Some(ResponsesShellEnvironment::Local { extra: Default::default() }),
        caller: direct_caller(), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::shell_call_container(
    json!({"type":"shell_call","call_id":"call_1","environment":{"type":"container_reference","container_id":"cntr_1"}}),
    ResponsesOutputItem::ShellCall(ResponsesShellCall {
        call_id: text("call_1"),
        environment: Some(ResponsesShellEnvironment::ContainerReference { container_id: "cntr_1".into(), extra: Default::default() }),
        ..Default::default()
    })
)]
#[case::shell_call_output(
    json!({"type":"shell_call_output","id":"sho_1","status":"completed","call_id":"call_1","max_output_length":100,"output":[
        {"outcome":{"type":"exit","exit_code":2},"stdout":"out","stderr":"err","created_by":"user_1"},
        {"outcome":{"type":"timeout"},"stdout":"","stderr":""}
    ],"caller":{"type":"program","caller_id":"prog_1"},"created_by":"user_1"}),
    ResponsesOutputItem::ShellCallOutput(ResponsesShellCallOutput {
        id: text("sho_1"), status: text("completed"), call_id: text("call_1"), max_output_length: Some(100),
        output: Some(vec![
            ResponsesShellOutputChunk {
                outcome: Some(ResponsesShellOutcome::Exit { exit_code: 2, extra: Default::default() }),
                stdout: text("out"), stderr: text("err"), created_by: text("user_1"), ..Default::default()
            },
            ResponsesShellOutputChunk {
                outcome: Some(ResponsesShellOutcome::Timeout { extra: Default::default() }),
                stdout: text(""), stderr: text(""), ..Default::default()
            },
        ]),
        caller: program_caller(), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::apply_patch_create(
    json!({"type":"apply_patch_call","id":"ap_1","status":"completed","call_id":"call_1","operation":{"type":"create_file","path":"a.txt","diff":"+a"},"caller":{"type":"direct"},"created_by":"user_1"}),
    ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
        id: text("ap_1"), status: text("completed"), call_id: text("call_1"),
        operation: Some(ResponsesApplyPatchOperation::CreateFile { path: "a.txt".into(), diff: "+a".into(), extra: Default::default() }),
        caller: direct_caller(), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::apply_patch_delete(
    json!({"type":"apply_patch_call","call_id":"call_1","operation":{"type":"delete_file","path":"a.txt"}}),
    ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
        call_id: text("call_1"),
        operation: Some(ResponsesApplyPatchOperation::DeleteFile { path: "a.txt".into(), extra: Default::default() }),
        ..Default::default()
    })
)]
#[case::apply_patch_update(
    json!({"type":"apply_patch_call","call_id":"call_1","operation":{"type":"update_file","path":"a.txt","diff":"-a\n+b"}}),
    ResponsesOutputItem::ApplyPatchCall(ResponsesApplyPatchCall {
        call_id: text("call_1"),
        operation: Some(ResponsesApplyPatchOperation::UpdateFile { path: "a.txt".into(), diff: "-a\n+b".into(), extra: Default::default() }),
        ..Default::default()
    })
)]
#[case::apply_patch_call_output(
    json!({"type":"apply_patch_call_output","id":"apo_1","status":"failed","call_id":"call_1","output":"conflict","caller":{"type":"direct"},"created_by":"user_1"}),
    ResponsesOutputItem::ApplyPatchCallOutput(ResponsesApplyPatchCallOutput {
        id: text("apo_1"), status: text("failed"), call_id: text("call_1"), output: text("conflict"),
        caller: direct_caller(), created_by: text("user_1"), ..Default::default()
    })
)]
#[case::mcp_approval_request(
    json!({"type":"mcp_approval_request","id":"apr_1","server_label":"s","name":"n","arguments":"{}"}),
    ResponsesOutputItem::McpApprovalRequest(ResponsesMcpApprovalRequest {
        id: text("apr_1"), server_label: text("s"), name: text("n"), arguments: text("{}"), ..Default::default()
    })
)]
#[case::mcp_approval_response(
    json!({"type":"mcp_approval_response","id":"aprr_1","approval_request_id":"apr_1","approve":false,"reason":"denied"}),
    ResponsesOutputItem::McpApprovalResponse(ResponsesMcpApprovalResponse {
        id: text("aprr_1"), approval_request_id: text("apr_1"), approve: Some(false), reason: text("denied"), ..Default::default()
    })
)]
fn documented_output_items_parse_into_typed_variants(
    #[case] wire: Value,
    #[case] expected: ResponsesOutputItem,
) {
    let item: ResponsesOutputItem = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(item, expected);
    assert_eq!(serde_json::to_value(item).unwrap(), wire);
}

#[rstest]
#[case::compaction(json!({"type":"compaction","id":"cmp_1","encrypted_content":"opaque"}))]
#[case::approval(json!({"type":"mcp_approval_request","id":"apr_1","server_label":"s","name":"n","arguments":"{}"}))]
#[case::shell(json!({"type":"shell_call","call_id":"call_1","action":{"commands":["ls"]}}))]
fn event_response_types_documented_output_items(#[case] item: Value) {
    let wire = json!({"output":[item.clone()]});
    let response: ResponsesEventResponse = serde_json::from_value(wire.clone()).unwrap();
    let Some([Recognized::Known(known)]) = response.output.as_deref() else {
        panic!("expected one typed item");
    };
    assert_eq!(
        known,
        &serde_json::from_value::<ResponsesOutputItem>(item).unwrap()
    );
    assert_eq!(serde_json::to_value(response).unwrap(), wire);
}

#[rstest]
#[case::unknown_caller(json!({"type":"function_call","caller":{"type":"future"}}))]
#[case::program_caller_missing_id(json!({"type":"shell_call","caller":{"type":"program"}}))]
#[case::untagged_caller(json!({"type":"custom_tool_call","caller":{}}))]
#[case::async_not_bool(json!({"type":"function_call","async":"yes"}))]
#[case::output_not_text_or_list(json!({"type":"function_call_output","output":{"text":"t"}}))]
#[case::unknown_input_content(json!({"type":"function_call_output","output":[{"type":"input_audio"}]}))]
#[case::input_text_missing_text(json!({"type":"custom_tool_call_output","output":[{"type":"input_text"}]}))]
#[case::input_image_missing_detail(json!({"type":"custom_tool_call_output","output":[{"type":"input_image","file_id":"f"}]}))]
#[case::bad_cache_mode(json!({"type":"custom_tool_call_output","output":[{"type":"input_text","text":"t","prompt_cache_breakpoint":{"mode":"implicit"}}]}))]
#[case::unknown_computer_action(json!({"type":"computer_call","action":{"type":"teleport"}}))]
#[case::click_missing_button(json!({"type":"computer_call","action":{"type":"click","x":1,"y":2}}))]
#[case::click_fractional_x(json!({"type":"computer_call","action":{"type":"click","button":"left","x":1.5,"y":2}}))]
#[case::drag_point_missing_y(json!({"type":"computer_call","actions":[{"type":"drag","path":[{"x":1}]}]}))]
#[case::keypress_keys_not_list(json!({"type":"computer_call","actions":[{"type":"keypress","keys":"enter"}]}))]
#[case::safety_check_not_object(json!({"type":"computer_call","pending_safety_checks":["sc_1"]}))]
#[case::unknown_screenshot_tag(json!({"type":"computer_call_output","output":{"type":"screenshot"}}))]
#[case::untagged_screenshot(json!({"type":"computer_call_output","output":{"file_id":"f"}}))]
#[case::tool_not_object(json!({"type":"tool_search_output","tools":["lookup"]}))]
#[case::unknown_local_shell_action(json!({"type":"local_shell_call","action":{"type":"spawn","command":[],"env":{}}}))]
#[case::local_shell_missing_env(json!({"type":"local_shell_call","action":{"type":"exec","command":["ls"]}}))]
#[case::local_shell_env_value(json!({"type":"local_shell_call","action":{"type":"exec","command":["ls"],"env":{"A":1}}}))]
#[case::shell_commands_not_list(json!({"type":"shell_call","action":{"commands":"ls"}}))]
#[case::unknown_shell_environment(json!({"type":"shell_call","environment":{"type":"container_auto"}}))]
#[case::container_missing_id(json!({"type":"shell_call","environment":{"type":"container_reference"}}))]
#[case::unknown_shell_outcome(json!({"type":"shell_call_output","output":[{"outcome":{"type":"killed"}}]}))]
#[case::exit_missing_code(json!({"type":"shell_call_output","output":[{"outcome":{"type":"exit"}}]}))]
#[case::negative_max_output(json!({"type":"shell_call_output","max_output_length":-1}))]
#[case::unknown_patch_operation(json!({"type":"apply_patch_call","operation":{"type":"rename_file","path":"a"}}))]
#[case::update_missing_diff(json!({"type":"apply_patch_call","operation":{"type":"update_file","path":"a"}}))]
#[case::approve_not_bool(json!({"type":"mcp_approval_response","approve":"true"}))]
#[case::compaction_content_not_string(json!({"type":"compaction","encrypted_content":7}))]
#[case::program_code_not_string(json!({"type":"program","code":["run()"]}))]
fn output_items_reject_malformed_typed_fields(#[case] wire: Value) {
    assert!(serde_json::from_value::<ResponsesOutputItem>(wire).is_err());
}
