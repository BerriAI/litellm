use litellm_traces::{CallEvidence, CallKey, DecodedSpan, ObservationType, decode_otlp};
use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest,
    common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value},
    trace::v1::{ResourceSpans, ScopeSpans, Span, span::Event},
};
use prost::Message;
use rstest::rstest;
use serde_json::{Value, json};

#[rstest::fixture]
fn span() -> Span {
    Span {
        trace_id: vec![1; 16],
        span_id: vec![2; 8],
        parent_span_id: vec![3; 8],
        name: "step".to_owned(),
        start_time_unix_nano: 1,
        end_time_unix_nano: 2,
        ..Default::default()
    }
}

fn recorded_attributes(attributes: &[(&str, &str)]) -> Vec<KeyValue> {
    attributes
        .iter()
        .map(|(key, value)| KeyValue {
            key: (*key).to_owned(),
            value: Some(AnyValue {
                value: Some(any_value::Value::StringValue((*value).to_owned())),
            }),
            ..Default::default()
        })
        .collect()
}

fn event(name: &str, attributes: &[(&str, &str)]) -> Event {
    Event {
        name: name.to_owned(),
        attributes: recorded_attributes(attributes),
        ..Default::default()
    }
}

fn decode(
    span: Span,
    scope: &str,
    attributes: &[(&str, &str)],
    events: Vec<Event>,
) -> Result<DecodedSpan, litellm_traces::Error> {
    let recorded = Span {
        attributes: recorded_attributes(attributes),
        events,
        ..span
    };
    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            scope_spans: vec![ScopeSpans {
                scope: Some(InstrumentationScope {
                    name: scope.to_owned(),
                    ..Default::default()
                }),
                spans: vec![recorded],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    Ok(decode_otlp(&request.encode_to_vec(), None)?
        .into_iter()
        .next()
        .unwrap())
}

#[rstest]
#[case::interaction("interaction", "user_prompt", "")]
#[case::model_context("llm_request", "new_context", "[USER]\n")]
fn native_claude_prompts_preserve_notification_text_and_user_role(
    span: Span,
    #[case] kind: &str,
    #[case] key: &str,
    #[case] prefix: &str,
) {
    let prompt = "<task-notification><summary>Quoted summary</summary><result>Keep this result</result></task-notification>\nExplain this example";
    let payload = format!("{prefix}{prompt}");
    let decoded = decode(
        span,
        "com.anthropic.claude_code.tracing",
        &[("span.type", kind), (key, &payload)],
        vec![],
    )
    .unwrap();
    let messages: Value = serde_json::from_str(&decoded.normalized.input).unwrap();
    assert_eq!(messages, json!([{"role": "user", "content": prompt}]));
}

#[rstest]
#[case::agent("agent", ObservationType::Agent)]
#[case::workflow("workflow", ObservationType::Chain)]
#[case::task("task", ObservationType::Chain)]
#[case::tool("tool", ObservationType::Tool)]
fn traceloop_extracts_entity_payloads_and_role(
    span: Span,
    #[case] kind: &str,
    #[case] expected: ObservationType,
) {
    let decoded = decode(
        span,
        "custom",
        &[
            ("traceloop.span.kind", kind),
            ("traceloop.entity.name", "lookup"),
            ("traceloop.entity.input", "query"),
            ("traceloop.entity.output", "result"),
            ("gen_ai.request.model", "fixture-model"),
            ("gen_ai.usage.prompt_tokens", "7"),
            ("gen_ai.usage.completion_tokens", "3"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
    assert_eq!(decoded.name, "lookup");
    assert_eq!(decoded.normalized.input, "query");
    assert_eq!(decoded.normalized.output, "result");
    assert_eq!(
        decoded.normalized.model.as_deref().unwrap_or_default(),
        decoded.attributes["gen_ai.request.model"]
    );
    assert_eq!(
        (
            decoded.normalized.input_tokens,
            decoded.normalized.output_tokens
        ),
        (7, 3)
    );
    assert!(
        decoded
            .consumed_attributes
            .contains(&"traceloop.entity.input")
    );
}

#[rstest]
#[case::generate("ai.generateText.doGenerate")]
#[case::stream("ai.streamText.doStream")]
fn vercel_preserves_messages_and_tool_calls(span: Span, #[case] operation: &str) {
    let calls = json!([{"toolCallId": "call-1", "toolName": "lookup", "args": {"q": "query"}}]);
    let decoded = decode(
        span,
        "ai",
        &[
            ("ai.operationId", operation),
            ("ai.model.id", "fixture-model"),
            (
                "ai.prompt.messages",
                r#"[{"role":"user","content":"query"}]"#,
            ),
            ("ai.response.text", "result"),
            ("ai.response.toolCalls", &calls.to_string()),
            ("ai.usage.promptTokens", "9"),
            ("ai.usage.completionTokens", "4"),
        ],
        vec![],
    )
    .unwrap();
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Llm);
    assert_eq!(decoded.normalized.input_preview, "query");
    assert_eq!(output[0]["content"], "result");
    assert_eq!(
        output[0]["tool_calls"],
        json!([{
            "id": calls[0]["toolCallId"], "name": calls[0]["toolName"], "arguments": calls[0]["args"],
        }])
    );
    assert_eq!(
        decoded.normalized.model.as_deref().unwrap_or_default(),
        decoded.attributes["ai.model.id"]
    );
    assert_eq!(
        (
            decoded.normalized.input_tokens,
            decoded.normalized.output_tokens
        ),
        (9, 4)
    );
}

#[rstest]
fn vercel_tool_records_arguments_result_and_identity(span: Span) {
    let decoded = decode(
        span,
        "ai",
        &[
            ("ai.operationId", "ai.toolCall"),
            ("ai.toolCall.name", "lookup"),
            ("ai.toolCall.id", "call-1"),
            ("ai.toolCall.args", r#"{"q":"query"}"#),
            ("ai.toolCall.result", "result"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Tool);
    assert_eq!(decoded.normalized.tool_call_id.as_deref(), Some("call-1"));
    assert_eq!(decoded.name, "lookup");
    assert_eq!(
        decoded.normalized.input,
        decoded.attributes["ai.toolCall.args"]
    );
    assert_eq!(decoded.normalized.output, "result");
}

#[rstest]
fn vercel_prompt_includes_system_and_user_messages(span: Span) {
    let decoded = decode(
        span,
        "ai",
        &[
            ("ai.operationId", "ai.generateText"),
            ("ai.prompt", r#"{"system":"instructions","prompt":"query"}"#),
            ("ai.response.object", r#"{"answer":42}"#),
        ],
        vec![],
    )
    .unwrap();
    let input: Value = serde_json::from_str(&decoded.normalized.input).unwrap();
    assert_eq!(
        input,
        json!([{"role":"system","content":"instructions"},{"role":"user","content":"query"}])
    );
    assert_eq!(
        decoded.normalized.output,
        decoded.attributes["ai.response.object"]
    );
}

#[rstest]
fn vercel_embedding_records_usage_and_input(span: Span) {
    let decoded = decode(
        span,
        "ai",
        &[
            ("ai.operationId", "ai.embed.doEmbed"),
            ("ai.value", "query"),
            ("ai.usage.tokens", "5"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(
        decoded.normalized.observation_type,
        ObservationType::Embedding
    );
    assert_eq!(decoded.normalized.input, "query");
    assert_eq!(decoded.normalized.input_tokens, 5);
}

#[rstest]
#[case::flat("gen_ai.prompt.2.role", "gen_ai.prompt.2.content")]
#[case::wrapped("gen_ai.prompt.2.message.role", "gen_ai.prompt.2.message.content")]
fn genai_indexed_messages_support_sparse_indices(
    span: Span,
    #[case] role: &str,
    #[case] content: &str,
) {
    let decoded = decode(
        span,
        "custom",
        &[
            (role, "user"),
            (content, "query"),
            ("gen_ai.completion.0.role", "assistant"),
            ("gen_ai.completion.0.content", "result"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.input_preview, "query");
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output, json!([{"role":"assistant","content":"result"}]));
}

#[rstest]
fn genai_message_events_extract_content_and_choice_tools(span: Span) {
    let calls = json!([{"id":"call-1","function":{"name":"lookup","arguments":"{}"}}]);
    let decoded = decode(
        span,
        "custom",
        &[],
        vec![
            event("unrelated", &[("content", "ignored")]),
            event("gen_ai.user.message", &[("content", "query")]),
            event(
                "gen_ai.choice",
                &[(
                    "gen_ai.event.content",
                    &json!({"message":{"content":"result","tool_calls":calls}}).to_string(),
                )],
            ),
        ],
    )
    .unwrap();
    assert_eq!(decoded.normalized.input_preview, "query");
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output[0]["content"], "result");
    assert_eq!(output[0]["tool_calls"], calls);
}

#[rstest]
#[case::single_message(
    json!({"role":"user","content":"hello"}),
    json!([{"role":"user","content":"hello"}])
)]
#[case::message_batch(
    json!([{"role":"user","content":"hello"},{"role":"assistant","content":"answer"}]),
    json!([{"role":"user","content":"hello"},{"role":"assistant","content":"answer"}])
)]
#[case::malformed_batch(
    json!([{"role":"user","content":"hello"},null]),
    json!([{"role":"user","content":"hello"},null])
)]
#[case::message_fields_are_not_a_message(
    json!([null,null,null,"user","hello",null,null,null,null]),
    json!([null,null,null,"user","hello",null,null,null,null])
)]
#[case::role_without_content(json!({"role":"user"}), json!({"role":"user"}))]
fn genai_message_payloads_preserve_non_conversations(
    span: Span,
    #[case] payload: Value,
    #[case] expected: Value,
) {
    let decoded = decode(
        span,
        "custom",
        &[("gen_ai.input.messages", &payload.to_string())],
        vec![],
    )
    .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&decoded.normalized.input).unwrap(),
        expected
    );
}

#[rstest]
#[case::all_messages("all_messages_events")]
#[case::events("events")]
fn logfire_splits_recorded_message_events(span: Span, #[case] key: &str) {
    let events = json!([
        null,
        {"event.name":7,"content":"ignored"},
        {"event.name":"unrelated","content":"ignored"},
        {"event.name":"gen_ai.user.message","content":"query"},
        {"event.name":"gen_ai.choice","message":{"role":"assistant","content":"result"}},
    ]);
    let decoded = decode(span, "pydantic-ai", &[(key, &events.to_string())], vec![]).unwrap();
    assert_eq!(decoded.normalized.input_preview, "query");
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output[0]["content"], "result");
    assert!(decoded.consumed_attributes.contains(&key));
}

#[rstest]
#[case::nested_wins(
    json!({"content":"root","role":"tool","message.content":"dotted","message.role":"user","message":{"content":"nested","role":"assistant"}}),
    json!([{"role":"assistant","content":"nested"}])
)]
#[case::nested_missing_uses_dotted(
    json!({"content":"root","role":"tool","message":{},"message.content":"dotted","message.role":"assistant"}),
    json!([{"role":"assistant","content":"dotted"}])
)]
#[case::null_message_uses_dotted(
    json!({"message":null,"content":"root","message.content":"dotted"}),
    json!([{"role":"assistant","content":"dotted"}])
)]
#[case::null_content_shadows_dotted(
    json!({"content":null,"message.content":"dotted"}),
    json!([{"role":"assistant","content":null,"tool_calls":null}])
)]
#[case::null_role_shadows_dotted(
    json!({"message":{"role":null,"content":"answer"},"message.role":"assistant"}),
    json!([{"role":null,"content":"answer","tool_calls":null}])
)]
#[case::null_calls_shadow_indexed(
    json!({"content":"answer","tool_calls":null,"tool_calls.0.function.name":"ignored"}),
    json!([{"role":"assistant","content":"answer"}])
)]
#[case::nested_indexed_calls(
    json!({"message":{"tool_calls.2.id":"call-2","tool_calls.2.function.name":"lookup","tool_calls.2.function.arguments":"{}"},"tool_calls.0.function.name":"ignored"}),
    json!([{"role":"assistant","content":"","tool_calls":[{"id":"call-2","name":"lookup","arguments":"{}"}]}])
)]
fn genai_event_envelopes_preserve_field_precedence(
    span: Span,
    #[case] payload: Value,
    #[case] expected: Value,
) {
    let decoded = decode(
        span,
        "custom",
        &[],
        vec![event(
            "gen_ai.choice",
            &[("gen_ai.event.content", &payload.to_string())],
        )],
    )
    .unwrap();
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output, expected);
}

#[rstest]
#[case::null(Value::Null)]
#[case::array(json!([]))]
#[case::number(json!(7))]
fn invalid_nested_event_messages_do_not_use_root_fields(span: Span, #[case] message: Value) {
    let payload =
        json!({"message":message,"content":"ignored","tool_calls.0.function.name":"ignored"});
    let decoded = decode(
        span,
        "custom",
        &[],
        vec![event(
            "gen_ai.choice",
            &[("gen_ai.event.content", &payload.to_string())],
        )],
    )
    .unwrap();
    assert!(decoded.normalized.output.is_empty());
}

#[rstest]
fn logfire_prompt_and_final_result_override_event_fallback(span: Span) {
    let decoded = decode(
        span,
        "logfire",
        &[
            ("prompt", "query"),
            ("final_result", "result"),
            ("events", "malformed"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.input, "query");
    assert_eq!(decoded.normalized.output, "result");
}

#[rstest]
#[case::vercel("ai", &[("ai.operationId", "ai.toolCall"), ("ai.prompt", "old"), ("ai.response.text", "old"), ("ai.usage.promptTokens", "99")])]
#[case::logfire("logfire", &[("prompt", "old"), ("final_result", "old")])]
fn modern_genai_fields_take_precedence(
    span: Span,
    #[case] scope: &str,
    #[case] legacy: &[(&str, &str)],
) {
    let attributes: Vec<_> = legacy
        .iter()
        .copied()
        .chain([
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.prompt", "query"),
            ("gen_ai.completion", "result"),
            ("gen_ai.usage.input_tokens", "0"),
            ("gen_ai.usage.prompt_tokens", "88"),
        ])
        .collect();
    let decoded = decode(span, scope, &attributes, vec![]).unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Llm);
    assert_eq!(decoded.normalized.input, "query");
    assert_eq!(decoded.normalized.output, "result");
    assert_eq!(decoded.normalized.input_tokens, 0);
}

#[rstest]
#[case::vercel("ai", &[("ai.operationId","ai.generateText"), ("ai.usage.promptTokens","-1")])]
#[case::deprecated("custom", &[("gen_ai.usage.prompt_tokens","4294967296")])]
fn legacy_token_counts_preserve_range_validation(
    span: Span,
    #[case] scope: &str,
    #[case] attributes: &[(&str, &str)],
) {
    assert!(decode(span, scope, attributes, vec![]).is_err());
}

#[rstest]
#[case::langsmith("langsmith.span.kind")]
#[case::openinference("openinference.span.kind")]
fn existing_formats_win_over_new_formats(span: Span, #[case] kind: &str) {
    let decoded = decode(
        span,
        "ai",
        &[
            (kind, "LLM"),
            ("ai.operationId", "ai.toolCall"),
            ("traceloop.span.kind", "tool"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Llm);
}

#[rstest]
#[case::messages(&[("gen_ai.input.messages", r#"[{"role":"user","content":"modern"}]"#)], "modern")]
#[case::indexed(&[("gen_ai.prompt.0.role", "user"), ("gen_ai.prompt.0.content", "indexed")], "indexed")]
#[case::events(&[], "event")]
fn genai_payload_precedence(
    span: Span,
    #[case] attributes: &[(&str, &str)],
    #[case] expected: &str,
) {
    let decoded = decode(
        span,
        "custom",
        attributes,
        vec![event("gen_ai.user.message", &[("content", "event")])],
    )
    .unwrap();
    assert_eq!(decoded.normalized.input_preview, expected);
}

#[rstest]
#[case::vercel("ai", &[("ai.operationId", "ai.generateText"), ("ai.prompt", "invalid-json"), ("ai.response.toolCalls", "invalid-json"), ("ai.response.text", "result")])]
#[case::traceloop("custom", &[("traceloop.entity.input", "invalid-json"), ("traceloop.entity.output", "result")])]
fn malformed_json_preserves_recorded_payloads(
    span: Span,
    #[case] scope: &str,
    #[case] attributes: &[(&str, &str)],
) {
    let decoded = decode(span, scope, attributes, vec![]).unwrap();
    assert_eq!(decoded.normalized.input, "invalid-json");
    assert_eq!(decoded.normalized.output, "result");
}

#[rstest]
fn unrelated_prompt_attributes_do_not_trigger_logfire(span: Span) {
    let decoded = decode(
        span,
        "custom",
        &[("prompt", "query"), ("events", "[]")],
        vec![],
    )
    .unwrap();
    assert!(decoded.normalized.input.is_empty());
    assert!(decoded.normalized.output.is_empty());
}

#[rstest]
#[case::embedding("embedding", ObservationType::Embedding)]
#[case::completion("completion", ObservationType::Llm)]
fn legacy_operation_names_are_classified(
    span: Span,
    #[case] operation: &str,
    #[case] expected: ObservationType,
) {
    let decoded = decode(
        span,
        "custom",
        &[("gen_ai.operation.name", operation)],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
}

#[rstest]
fn langsmith_kind_preserves_genai_indexed_payloads(span: Span) {
    let decoded = decode(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "llm"),
            ("gen_ai.prompt.0.role", "user"),
            ("gen_ai.prompt.0.content", "query"),
            ("gen_ai.completion.0.role", "assistant"),
            ("gen_ai.completion.0.content", "result"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Llm);
    assert_eq!(decoded.normalized.input_preview, "query");
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output[0]["content"], "result");
}

#[rstest]
fn genai_choice_events_support_flattened_tool_calls(span: Span) {
    let decoded = decode(
        span,
        "custom",
        &[],
        vec![event(
            "gen_ai.choice",
            &[
                ("message.role", "assistant"),
                ("tool_calls.2.id", "call-1"),
                ("tool_calls.2.function.name", "lookup"),
                ("tool_calls.2.function.arguments", "{}"),
            ],
        )],
    )
    .unwrap();
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(output[0]["role"], "assistant");
    assert_eq!(
        output[0]["tool_calls"],
        json!([{"id":"call-1","name":"lookup","arguments":"{}"}])
    );
}

#[rstest]
fn genai_indexed_tool_only_completion_keeps_calls(span: Span) {
    let decoded = decode(
        span,
        "custom",
        &[
            ("gen_ai.completion.0.role", "assistant"),
            ("gen_ai.completion.0.tool_calls.0.id", "call-1"),
            ("gen_ai.completion.0.tool_calls.0.function.name", "lookup"),
            ("gen_ai.completion.0.tool_calls.0.function.arguments", "{}"),
        ],
        vec![],
    )
    .unwrap();
    let output: Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(
        output[0]["tool_calls"],
        json!([{"id":"call-1","name":"lookup","arguments":"{}"}])
    );
}

#[rstest]
#[case::embedding("embedding", ObservationType::Embedding)]
#[case::chat("chat", ObservationType::Llm)]
fn traceloop_request_type_is_used_without_entity_kind(
    span: Span,
    #[case] request: &str,
    #[case] expected: ObservationType,
) {
    let decoded = decode(
        span,
        "custom",
        &[
            ("traceloop.entity.name", "request"),
            ("llm.request.type", request),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
}

#[rstest]
fn absent_normalized_identity_fields_stay_absent(span: Span) {
    let decoded = decode(span, "custom", &[], vec![]).unwrap();
    assert_eq!(decoded.normalized.agent_name, None);
    assert_eq!(decoded.normalized.framework, None);
    assert_eq!(decoded.normalized.model, None);
    assert_eq!(decoded.normalized.tool_call_id, None);
    assert_eq!(
        decoded.normalized.calls,
        litellm_traces::CallEvidence::Unknown
    );
}

#[rstest]
#[case::provider(json!({"id": "provider-response"}), Some("provider-response"))]
#[case::llamaindex(json!({"message": {"role": "assistant", "content": "answer"}, "raw": {"id": "wrapped-response"}}), Some("wrapped-response"))]
#[case::missing(json!({"raw": {"usage": {"total_tokens": 8}}}), None)]
#[case::invalid(json!({"raw": {"id": 123}}), None)]
fn openinference_provider_response_identity(
    span: Span,
    #[case] response: Value,
    #[case] id: Option<&str>,
) {
    let decoded = decode(
        span,
        "openinference.instrumentation.llama_index",
        &[
            ("openinference.span.kind", "LLM"),
            ("output.value", &response.to_string()),
        ],
        vec![],
    )
    .unwrap();
    let expected = id.map_or(CallEvidence::Unknown, |id| {
        CallEvidence::Complete(std::collections::BTreeSet::from([
            CallKey::ProviderResponse(id.to_owned()),
        ]))
    });
    assert_eq!(decoded.normalized.calls, expected);
}

#[rstest]
#[case::unknown("custom", "gen_ai.response.id", CallKey::ProviderResponse("id".into()))]
#[case::gateway("custom", "litellm.call_id", CallKey::LiteLlmRequest("id".into()))]
#[case::other_format("langsmith", "litellm.call_id", CallKey::LiteLlmRequest("id".into()))]
fn generic_ids_do_not_prove_call_completeness(
    span: Span,
    #[case] scope: &str,
    #[case] attribute: &str,
    #[case] key: CallKey,
) {
    let decoded = decode(span, scope, &[(attribute, "id")], vec![]).unwrap();
    assert_eq!(
        decoded.normalized.calls,
        CallEvidence::Partial(std::collections::BTreeSet::from([key]))
    );
}

#[rstest]
#[case::chat("chat", true)]
#[case::text_completion("text_completion", true)]
#[case::generate_content("generate_content", true)]
#[case::agent("invoke_agent", false)]
#[case::tool("execute_tool", false)]
fn genai_model_operations_with_a_response_id_are_complete_calls(
    span: Span,
    #[case] operation: &str,
    #[case] complete: bool,
) {
    let decoded = decode(
        span,
        "custom",
        &[
            ("gen_ai.operation.name", operation),
            ("gen_ai.response.id", "chatcmpl-1"),
        ],
        vec![],
    )
    .unwrap();
    let keys = std::collections::BTreeSet::from([CallKey::ProviderResponse("chatcmpl-1".into())]);
    let expected = if complete {
        CallEvidence::Complete(keys)
    } else {
        CallEvidence::Partial(keys)
    };
    assert_eq!(decoded.normalized.calls, expected);
}

#[rstest]
fn transport_contract_keeps_independent_call_ids(span: Span) {
    let decoded = decode(
        span,
        "opentelemetry.instrumentation.httpx",
        &[
            ("litellm.call_id", "gateway"),
            ("gen_ai.response.id", "response"),
        ],
        vec![],
    )
    .unwrap();
    assert_eq!(
        decoded.normalized.calls,
        CallEvidence::Complete(std::collections::BTreeSet::from([
            CallKey::Transport,
            CallKey::LiteLlmRequest("gateway".into()),
            CallKey::ProviderResponse("response".into()),
        ]))
    );
}

#[rstest]
#[case::request("litellm.gateway.client", "gateway.request", "true", "POST", true)]
#[case::unrelated_scope("custom", "gateway.request", "true", "POST", false)]
#[case::unrelated_span("litellm.gateway.client", "step", "true", "POST", false)]
#[case::missing_contract("litellm.gateway.client", "gateway.request", "", "POST", false)]
#[case::disabled_contract("litellm.gateway.client", "gateway.request", "false", "POST", false)]
#[case::unrelated_method("litellm.gateway.client", "gateway.request", "true", "GET", false)]
fn gateway_attempt_contract_requires_recorded_request_boundary(
    span: Span,
    #[case] scope: &str,
    #[case] name: &str,
    #[case] attempt: &str,
    #[case] method: &str,
    #[case] complete: bool,
) {
    let decoded = decode(
        Span {
            name: name.into(),
            ..span
        },
        scope,
        &[
            ("litellm.gateway.attempt", attempt),
            ("http.request.method", method),
            ("litellm.call_id", "gateway"),
        ],
        vec![],
    )
    .unwrap();
    let gateway = CallKey::LiteLlmRequest("gateway".into());
    assert_eq!(
        decoded.normalized.calls,
        if complete {
            CallEvidence::Complete(std::collections::BTreeSet::from([
                CallKey::GatewayAttempt,
                gateway,
            ]))
        } else {
            CallEvidence::Partial(std::collections::BTreeSet::from([gateway]))
        }
    );
    if complete {
        assert_eq!(
            decoded.normalized.observation_type,
            ObservationType::Framework
        );
    }
}

#[rstest]
#[case::both(true, true)]
#[case::input_only(true, false)]
#[case::output_only(false, true)]
fn langsmith_consumption_follows_selected_payloads(
    span: Span,
    #[case] legacy_input: bool,
    #[case] legacy_output: bool,
) {
    let decoded = decode(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "chain"),
            (
                "gen_ai.input.messages",
                r#"[{"role":"user","content":"modern input"}]"#,
            ),
            (
                "gen_ai.output.messages",
                r#"[{"role":"assistant","content":"modern output"}]"#,
            ),
            (
                "gen_ai.prompt",
                if legacy_input { "legacy input" } else { "" },
            ),
            (
                "gen_ai.completion",
                if legacy_output { "legacy output" } else { "" },
            ),
        ],
        vec![],
    )
    .unwrap();
    for (legacy, modern, selected, payload, expected) in [
        (
            "gen_ai.prompt",
            "gen_ai.input.messages",
            legacy_input,
            &decoded.normalized.input,
            "legacy input",
        ),
        (
            "gen_ai.completion",
            "gen_ai.output.messages",
            legacy_output,
            &decoded.normalized.output,
            "legacy output",
        ),
    ] {
        assert_eq!(decoded.consumed_attributes.contains(&legacy), selected);
        assert_eq!(decoded.consumed_attributes.contains(&modern), !selected);
        if selected {
            assert_eq!(payload, expected);
        } else {
            assert!(serde_json::from_str::<Value>(payload).unwrap().is_array());
        }
    }
}

#[rstest]
#[case::with_output_messages(&[
    ("langsmith.span.kind", "llm"),
    ("gen_ai.operation.name", "chat"),
    ("gen_ai.response.id", "chatcmpl-1"),
    (
        "gen_ai.output.messages",
        r#"[{"role":"assistant","parts":[{"type":"text","content":"hi"}]}]"#,
    ),
])]
#[case::without_output_messages(&[
    ("langsmith.span.kind", "llm"),
    ("gen_ai.operation.name", "chat"),
    ("gen_ai.response.id", "chatcmpl-1"),
])]
fn langsmith_response_id_is_complete_without_legacy_payloads(
    span: Span,
    #[case] attributes: &[(&str, &str)],
) {
    let decoded = decode(span, "langsmith", attributes, vec![]).unwrap();
    assert_eq!(
        decoded.normalized.calls,
        CallEvidence::Complete(std::collections::BTreeSet::from([
            CallKey::ProviderResponse("chatcmpl-1".into()),
        ]))
    );
}

#[rstest]
#[case::langsmith("langsmith", "langsmith.span.kind", "llm")]
#[case::logfire("logfire", "events", "[]")]
#[case::traceloop("custom", "traceloop.span.kind", "llm")]
#[case::vercel("ai", "ai.operationId", "ai.generateText")]
fn convention_markers_keep_genai_call_evidence(
    span: Span,
    #[case] scope: &str,
    #[case] marker: &str,
    #[case] marker_value: &str,
) {
    let attributes = [
        ("gen_ai.operation.name", "chat"),
        ("gen_ai.response.id", "chatcmpl-1"),
        ("gen_ai.request.model", "fixture-model"),
        (
            "gen_ai.output.messages",
            r#"[{"role":"assistant","parts":[{"type":"text","content":"hi"}]}]"#,
        ),
    ];
    let plain = decode(span.clone(), "custom", &attributes, vec![]).unwrap();
    let marked_attributes = attributes
        .into_iter()
        .chain([(marker, marker_value)])
        .collect::<Vec<_>>();
    let marked = decode(span, scope, &marked_attributes, vec![]).unwrap();
    assert_eq!(marked.normalized.calls, plain.normalized.calls);
}

#[rstest]
#[case::request("req_native", CallKey::ProviderRequest("req_native".into()))]
#[case::legacy_message("msg_legacy", CallKey::ProviderResponse("msg_legacy".into()))]
fn native_claude_preserves_the_provider_id_family(
    span: Span,
    #[case] id: &str,
    #[case] key: CallKey,
) {
    let native = Span {
        name: "claude_code.llm_request".into(),
        ..span
    };
    let decoded = decode(
        native,
        "com.anthropic.claude_code.tracing",
        &[("gen_ai.response.id", id)],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(
        decoded.normalized.calls,
        CallEvidence::Complete(std::collections::BTreeSet::from([key]))
    );
}
