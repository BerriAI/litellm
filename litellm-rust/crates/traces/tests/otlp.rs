use litellm_traces::decode_otlp;
use litellm_traces::{ObservationType, Shared};
use opentelemetry_proto::tonic::trace::v1::Span;
use rstest::rstest;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"
);

#[rstest]
#[case::root(include_bytes!("fixtures/query_root.json"), ObservationType::Agent, 0, 0)]
#[case::children(include_bytes!("fixtures/query_children.json"), ObservationType::Llm, 12, 6)]
#[case::alternate(include_bytes!("fixtures/query_alternate.json"), ObservationType::Agent, 0, 0)]
#[case::other_team(include_bytes!("fixtures/query_other_team.json"), ObservationType::Agent, 0, 0)]
fn query_fixtures_decode_and_normalize(
    #[case] body: &[u8],
    #[case] observation_type: ObservationType,
    #[case] input_tokens: u32,
    #[case] output_tokens: u32,
) {
    let spans = decode_otlp(body, Some("application/json")).unwrap();
    let first = &spans[0];
    assert_eq!(first.normalized.observation_type, observation_type);
    assert_eq!(first.normalized.input_tokens, input_tokens);
    assert_eq!(first.normalized.output_tokens, output_tokens);
    assert!(
        spans
            .iter()
            .all(|span| span.resource_attributes["service.name"] == "fixture")
    );
}

#[rstest]
#[case::json(FIXTURE, Some("application/json"))]
fn decodes_neutral_spans(#[case] body: &[u8], #[case] content_type: Option<&str>) {
    let spans = decode_otlp(body, content_type).expect("valid OTLP export");
    assert_eq!(spans.len(), 6);
    assert_eq!(spans[0].trace_id, "4bad42b84e9de3ba46fc870185f8f023");
    assert_eq!(spans[0].resource_attributes["service.name"], "agent-demo");
    assert_eq!(spans[0].scope_name.as_ref(), "langsmith");
    assert!(
        spans
            .iter()
            .any(|span| span.attributes.contains_key("gen_ai.prompt"))
    );
}

#[rstest]
fn accepts_trace_larger_than_eight_mib(mut span: opentelemetry_proto::tonic::trace::v1::Span) {
    use prost::Message;

    span.name = "x".repeat(9 * 1024 * 1024);
    let body = request_with(span).encode_to_vec();
    let decoded = decode_otlp(&body, None).expect("16 MiB default accepts a 9 MiB trace");
    assert_eq!(decoded[0].name.len(), 9 * 1024 * 1024);
}

#[rstest]
fn rejects_invalid_payload() {
    assert!(decode_otlp(b"not protobuf", None).is_err());
}

#[rstest]
fn decoder_does_not_enforce_the_http_body_limit() {
    let body = format!("{{\"ignored\":\"{}\"}}", "x".repeat(16 * 1024 * 1024 + 1));
    assert!(
        decode_otlp(body.as_bytes(), Some("application/json"))
            .unwrap()
            .is_empty()
    );
}

fn request_with(
    span: opentelemetry_proto::tonic::trace::v1::Span,
) -> opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest {
    use opentelemetry_proto::tonic::{
        collector::trace::v1::ExportTraceServiceRequest,
        trace::v1::{ResourceSpans, ScopeSpans},
    };
    ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            scope_spans: vec![ScopeSpans {
                spans: vec![span],
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

#[rstest::fixture]
fn span() -> opentelemetry_proto::tonic::trace::v1::Span {
    opentelemetry_proto::tonic::trace::v1::Span {
        trace_id: vec![1; 16],
        span_id: vec![2; 8],
        start_time_unix_nano: 1,
        end_time_unix_nano: 2,
        ..Default::default()
    }
}

#[rstest]
fn standard_json_and_protobuf_preserve_the_same_identifiers(
    span: opentelemetry_proto::tonic::trace::v1::Span,
) {
    use prost::Message;
    let request = request_with(span);
    let json = serde_json::to_vec(&request).unwrap();
    let binary = request.encode_to_vec();
    let json_spans = decode_otlp(&json, Some("application/json; charset=utf-8")).unwrap();
    let binary_spans = decode_otlp(&binary, Some("application/x-protobuf")).unwrap();
    assert_eq!(
        serde_json::to_value(&json_spans).unwrap(),
        serde_json::to_value(&binary_spans).unwrap()
    );
    assert_eq!(json_spans[0].trace_id, "01".repeat(16));
    assert_eq!(json_spans[0].span_id, "02".repeat(8));
}

#[rstest]
#[case::json("APPLICATION/JSON; charset=utf-8", b"{}")]
#[case::protobuf("application/x-protobuf; charset=binary", b"")]
#[case::protobuf_alias("APPLICATION/PROTOBUF", b"")]
fn supported_content_types_select_the_decoder(#[case] content_type: &str, #[case] body: &[u8]) {
    assert!(decode_otlp(body, Some(content_type)).is_ok());
}

#[rstest]
#[case::missing_content_type(None)]
#[case::unsupported_content_type(Some("text/plain"))]
fn content_type_defaults_to_protobuf_and_rejects_unknown_values(
    #[case] content_type: Option<&str>,
) {
    let result = decode_otlp(b"", content_type);
    assert_eq!(result.is_ok(), content_type.is_none());
}

#[rstest]
#[case::short_trace(vec![1; 15], vec![2;8], 1, 2)]
#[case::zero_trace(vec![0; 16], vec![2;8], 1, 2)]
#[case::short_span(vec![1; 16], vec![2;7], 1, 2)]
#[case::timestamp_overflow(vec![1;16], vec![2;8], i64::MAX as u64 + 1, i64::MAX as u64 + 1)]
#[case::negative_duration(vec![1;16], vec![2;8], 3, 2)]
fn rejects_ids_and_timestamps_that_cannot_be_stored(
    #[case] trace_id: Vec<u8>,
    #[case] span_id: Vec<u8>,
    #[case] start: u64,
    #[case] end: u64,
) {
    use prost::Message;
    let span = opentelemetry_proto::tonic::trace::v1::Span {
        trace_id,
        span_id,
        start_time_unix_nano: start,
        end_time_unix_nano: end,
        ..Default::default()
    };
    assert!(matches!(
        decode_otlp(&request_with(span).encode_to_vec(), None),
        Err(litellm_traces::Error::InvalidPayload)
    ));
}

#[rstest]
fn resource_fanout_shares_one_allocation(span: opentelemetry_proto::tonic::trace::v1::Span) {
    use opentelemetry_proto::tonic::{
        common::v1::{AnyValue, KeyValue, any_value::Value},
        resource::v1::Resource,
    };
    use prost::Message;
    let mut request = request_with(span.clone());
    request.resource_spans[0].resource = Some(Resource {
        attributes: vec![KeyValue {
            key: "shared".into(),
            value: Some(AnyValue {
                value: Some(Value::StringValue("x".repeat(16 * 1024))),
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    request.resource_spans[0].scope_spans[0].spans = vec![span; 1024];
    let second_scope = request.resource_spans[0].scope_spans[0].clone();
    request.resource_spans[0].scope_spans.push(second_scope);
    request
        .resource_spans
        .push(request.resource_spans[0].clone());
    let body = request.encode_to_vec();
    let decoded = decode_otlp(&body, None).expect("shared resources do not expand with span count");
    assert_eq!(decoded.len(), 4096);
    assert!(decoded[..2048].iter().all(|span| {
        Shared::shares_storage_with(&span.resource_attributes, &decoded[0].resource_attributes)
    }));
    assert!(!Shared::shares_storage_with(
        &decoded[0].resource_attributes,
        &decoded[2048].resource_attributes
    ));
    assert_eq!(
        *decoded[0].resource_attributes,
        *decoded[2048].resource_attributes
    );
}

#[rstest]
fn nested_values_are_serialized_once(span: opentelemetry_proto::tonic::trace::v1::Span) {
    use opentelemetry_proto::tonic::common::v1::{
        AnyValue, ArrayValue, KeyValue, any_value::Value,
    };
    use prost::Message;
    let nested = (0..8).fold(
        AnyValue {
            value: Some(Value::StringValue("quoted \"value\"".into())),
        },
        |child, _| AnyValue {
            value: Some(Value::ArrayValue(ArrayValue {
                values: vec![child],
            })),
        },
    );
    let mut request = request_with(span);
    request.resource_spans[0].scope_spans[0].spans[0].attributes = vec![KeyValue {
        key: "nested".into(),
        value: Some(nested),
        ..Default::default()
    }];
    let spans = decode_otlp(&request.encode_to_vec(), None).unwrap();
    let expected = (0..8).fold(serde_json::json!("quoted \"value\""), |child, _| {
        serde_json::json!([child])
    });
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&spans[0].attributes["nested"]).unwrap(),
        expected
    );
    assert!(spans[0].attributes["nested"].len() < 64);
}

#[rstest]
#[case::nesting(format!("{}0{}", "[".repeat(40), "]".repeat(40)).into_bytes())]
#[case::nodes(format!("[{}]", vec!["0"; 65537].join(",")).into_bytes())]
fn rejects_json_structure_before_building_a_tree(#[case] body: Vec<u8>) {
    assert!(matches!(
        decode_otlp(&body, Some("application/json")),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
#[case::depth(40, 1)]
#[case::nodes(0, 65537)]
fn protobuf_preflight_rejects_expansion_before_prost_allocates(
    span: opentelemetry_proto::tonic::trace::v1::Span,
    #[case] depth: usize,
    #[case] count: usize,
) {
    use opentelemetry_proto::tonic::common::v1::{
        AnyValue, ArrayValue, KeyValue, any_value::Value,
    };
    use prost::Message;
    let value = (0..depth).fold(
        AnyValue {
            value: Some(Value::BoolValue(true)),
        },
        |child, _| AnyValue {
            value: Some(Value::ArrayValue(ArrayValue {
                values: vec![child],
            })),
        },
    );
    let mut request = request_with(span);
    request.resource_spans[0].scope_spans[0].spans[0].attributes = vec![KeyValue {
        key: "deep".into(),
        value: Some(value),
        ..Default::default()
    }];
    request.resource_spans = vec![request.resource_spans[0].clone(); count];
    let body = request.encode_to_vec();
    assert!(matches!(
        decode_otlp(&body, None),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
fn scope_fanout_shares_name_and_version(span: opentelemetry_proto::tonic::trace::v1::Span) {
    use opentelemetry_proto::tonic::common::v1::InstrumentationScope;
    use prost::Message;
    let mut request = request_with(span.clone());
    request.resource_spans[0].scope_spans[0].scope = Some(InstrumentationScope {
        name: "n".repeat(16 * 1024),
        version: "v".repeat(16 * 1024),
        ..Default::default()
    });
    request.resource_spans[0].scope_spans[0].spans = vec![span; 1024];
    let decoded = decode_otlp(&request.encode_to_vec(), None).unwrap();
    assert!(
        decoded
            .iter()
            .all(|span| Shared::shares_storage_with(&span.scope_name, &decoded[0].scope_name))
    );
    assert!(
        decoded.iter().all(|span| Shared::shares_storage_with(
            &span.scope_version,
            &decoded[0].scope_version
        ))
    );
    assert_eq!(decoded[0].scope_name.len(), 16 * 1024);
    assert_eq!(decoded[0].scope_version.len(), 16 * 1024);
}

#[rstest]
fn unique_attribute_expansion_still_respects_decoded_budget(
    span: opentelemetry_proto::tonic::trace::v1::Span,
) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use prost::Message;
    let mut request = request_with(span.clone());
    request.resource_spans[0].scope_spans[0].spans = (0..1024)
        .map(|index| {
            let mut span = span.clone();
            span.attributes = vec![KeyValue {
                key: "unique".into(),
                value: Some(AnyValue {
                    value: Some(Value::StringValue(format!(
                        "{index:04}{}",
                        "x".repeat(16_300)
                    ))),
                }),
                ..Default::default()
            }];
            span
        })
        .collect();
    let body = request.encode_to_vec();
    assert!(body.len() < 16 * 1024 * 1024);
    assert!(matches!(
        decode_otlp(&body, None),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
fn escaped_attribute_expansion_is_bounded_below_four_mib(
    span: opentelemetry_proto::tonic::trace::v1::Span,
) {
    use opentelemetry_proto::tonic::common::v1::{
        AnyValue, ArrayValue, KeyValue, any_value::Value,
    };
    use prost::Message;
    let mut request = request_with(span);
    request.resource_spans[0].scope_spans[0].spans[0].attributes = vec![KeyValue {
        key: "escaped".into(),
        value: Some(AnyValue {
            value: Some(Value::ArrayValue(ArrayValue {
                values: vec![AnyValue {
                    value: Some(Value::StringValue("\0".repeat(3 * 1024 * 1024))),
                }],
            })),
        }),
        ..Default::default()
    }];
    let body = request.encode_to_vec();
    assert!(body.len() < 4 * 1024 * 1024);
    assert!(matches!(
        decode_otlp(&body, None),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
fn normalizes_langsmith_fixture() {
    let spans = decode_otlp(FIXTURE, Some("application/json")).expect("valid OTLP export");
    let llm = spans
        .iter()
        .find(|span| span.name == "ChatOpenAI")
        .expect("LLM span");
    assert_eq!(llm.normalized.observation_type, ObservationType::Llm);
    assert_eq!(llm.normalized.agent_name, "deep_research_agent");
    assert_eq!(llm.normalized.model, "claude-sonnet-4-5");
    assert_eq!(
        (llm.normalized.input_tokens, llm.normalized.output_tokens),
        (3332, 467)
    );
    assert_eq!(
        llm.normalized.litellm_request_id,
        "chatcmpl-4077bb36-9380-4a3b-9481-245700cef09a"
    );
    let input: serde_json::Value =
        serde_json::from_str(&llm.normalized.input).expect("message input");
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "user");
    let output: serde_json::Value =
        serde_json::from_str(&llm.normalized.output).expect("message output");
    assert_eq!(output["role"], "assistant");
    assert!(output["tool_calls"][0]["name"].is_string());
    assert!(output["tool_calls"][0]["id"].is_string());
    assert_eq!(output["tool_calls"][0]["type"], "tool_call");
    let root = spans
        .iter()
        .find(|span| span.name == "deep_research_agent")
        .expect("root span");
    assert_eq!(root.normalized.observation_type, ObservationType::Agent);
    assert_eq!(
        root.normalized.input,
        "[{\"role\": \"user\", \"content\": \"Should we store OTEL agent spans in ClickHouse or Postgres at 50k spans/sec?\"}]"
    );
    let tool = spans
        .iter()
        .find(|span| span.name == "task")
        .expect("tool span");
    assert_eq!(tool.normalized.observation_type, ObservationType::Tool);
    assert!(tool.normalized.output.starts_with("Based on my research"));
}

fn decode_normalization(
    span: Span,
    scope: &str,
    attributes: &[(&str, &str)],
) -> Result<litellm_traces::DecodedSpan, litellm_traces::Error> {
    use opentelemetry_proto::tonic::{
        collector::trace::v1::ExportTraceServiceRequest,
        common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value::Value},
        trace::v1::{ResourceSpans, ScopeSpans},
    };
    use prost::Message;

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            scope_spans: vec![ScopeSpans {
                scope: Some(InstrumentationScope {
                    name: scope.to_owned(),
                    ..Default::default()
                }),
                spans: vec![Span {
                    attributes: attributes
                        .iter()
                        .map(|(key, value)| KeyValue {
                            key: (*key).to_owned(),
                            value: Some(AnyValue {
                                value: Some(Value::StringValue((*value).to_owned())),
                            }),
                            ..Default::default()
                        })
                        .collect(),
                    ..span
                }],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    decode_otlp(&request.encode_to_vec(), None)
        .map(|spans| spans.into_iter().next().expect("one synthetic span"))
}

#[rstest]
#[case::agent("invoke_agent", false, ObservationType::Agent)]
#[case::chat("chat", false, ObservationType::Llm)]
#[case::completion("text_completion", false, ObservationType::Llm)]
#[case::content("generate_content", false, ObservationType::Llm)]
#[case::tool("execute_tool", false, ObservationType::Tool)]
#[case::unknown_root("unknown", true, ObservationType::Agent)]
#[case::unknown_child("unknown", false, ObservationType::Chain)]
#[case::missing_root("", true, ObservationType::Agent)]
#[case::missing_child("", false, ObservationType::Chain)]
fn genai_operations_and_parentage_classify_spans(
    span: Span,
    #[case] operation: &str,
    #[case] root: bool,
    #[case] expected: ObservationType,
) {
    let decoded = decode_normalization(
        Span {
            parent_span_id: if root { vec![] } else { vec![3; 8] },
            ..span
        },
        "",
        &[("gen_ai.operation.name", operation)],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
}

#[rstest]
#[case::primary("request-model", "messages-in", "messages-out", ["request-model", "messages-in", "messages-out"])]
#[case::fallback("", "", "", ["response-model", "tool-in", "tool-out"])]
#[case::independent_fallback("request-model", "", "messages-out", ["request-model", "tool-in", "messages-out"])]
fn genai_fields_and_consumed_attributes_follow_the_same_fallback(
    span: Span,
    #[case] model: &str,
    #[case] input: &str,
    #[case] output: &str,
    #[case] expected: [&str; 3],
) {
    let decoded = decode_normalization(
        span,
        "",
        &[
            ("gen_ai.request.model", model),
            ("gen_ai.response.model", "response-model"),
            ("gen_ai.input.messages", input),
            ("gen_ai.output.messages", output),
            ("gen_ai.tool.call.arguments", "tool-in"),
            ("gen_ai.tool.call.result", "tool-out"),
            ("gen_ai.agent.name", "test-agent"),
            ("gen_ai.response.id", "response-1"),
        ],
    )
    .unwrap();
    let fields = &decoded.normalized;
    assert_eq!(
        [
            fields.model.as_str(),
            fields.input.as_str(),
            fields.output.as_str()
        ],
        expected
    );
    assert_eq!(fields.agent_name, "test-agent");
    assert_eq!(fields.litellm_request_id, "response-1");
    assert_eq!(
        fields.input,
        decoded.attributes[decoded.consumed_attributes[0]]
    );
    assert_eq!(
        fields.output,
        decoded.attributes[decoded.consumed_attributes[1]]
    );
}

#[rstest]
#[case::specific(&[("llm.token_count.prompt", "5"), ("llm.token_count.completion", "9")], 5, 9)]
#[case::fallback(&[], 17, 23)]
#[case::mixed(&[("llm.token_count.prompt", "5")], 5, 23)]
#[case::empty_specific(&[("llm.token_count.prompt", "")], 0, 23)]
fn openinference_fields_override_genai_and_usage_falls_back_per_field(
    span: Span,
    #[case] token_attributes: &[(&str, &str)],
    #[case] input_tokens: u32,
    #[case] output_tokens: u32,
) {
    let attributes = [
        ("openinference.span.kind", "lLm"),
        ("gen_ai.operation.name", "execute_tool"),
        ("llm.model_name", "inference-model"),
        ("gen_ai.request.model", "other-model"),
        ("agent.name", "inference-agent"),
        ("input.value", "inference-input"),
        ("output.value", "inference-output"),
        ("gen_ai.input.messages", "other-input"),
        ("gen_ai.output.messages", "other-output"),
        ("gen_ai.usage.input_tokens", "17"),
        ("gen_ai.usage.output_tokens", "23"),
    ];
    let combined = attributes
        .iter()
        .chain(token_attributes)
        .copied()
        .collect::<Vec<_>>();
    let decoded = decode_normalization(span, "", &combined).unwrap();
    let fields = &decoded.normalized;
    assert_eq!(fields.observation_type, ObservationType::Llm);
    assert_eq!(fields.model, "inference-model");
    assert_eq!(fields.agent_name, "inference-agent");
    assert_eq!(fields.input, "inference-input");
    assert_eq!(fields.output, "inference-output");
    assert_eq!(
        (fields.input_tokens, fields.output_tokens),
        (input_tokens, output_tokens)
    );
    assert_eq!(decoded.consumed_attributes, ["input.value", "output.value"]);
}

#[rstest]
#[case::scope("langsmith", &[], ObservationType::Agent)]
#[case::attribute("other", &[("langsmith.span.kind", "llm")], ObservationType::Llm)]
fn langsmith_dispatch_overrides_other_conventions(
    span: Span,
    #[case] scope: &str,
    #[case] convention_attributes: &[(&str, &str)],
    #[case] observation_type: ObservationType,
) {
    let attributes = [
        ("openinference.span.kind", "TOOL"),
        ("gen_ai.operation.name", "execute_tool"),
        ("langsmith.metadata.lc_agent_name", "test-agent"),
        (
            "gen_ai.prompt",
            r#"{"messages":[{"type":"human","content":"hello"}]}"#,
        ),
        ("gen_ai.completion", "{}"),
        ("input.value", "other-input"),
    ];
    let combined = attributes
        .iter()
        .chain(convention_attributes)
        .copied()
        .collect::<Vec<_>>();
    let decoded = decode_normalization(span, scope, &combined).unwrap();
    assert_eq!(decoded.normalized.observation_type, observation_type);
    assert_eq!(decoded.normalized.agent_name, "test-agent");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&decoded.normalized.input).unwrap(),
        serde_json::json!([{"role": "user", "content": "hello"}]),
    );
    assert_eq!(
        decoded.consumed_attributes,
        ["gen_ai.prompt", "gen_ai.completion"]
    );
}

#[rstest]
#[case::flat(r#"{"messages":[{"type":"human","content":"hello"}]}"#)]
#[case::nested(r#"{"messages":[[{"kwargs":{"type":"human","content":"hello"}}],[{"type":"human","content":"ignored batch"}]]}"#)]
fn langsmith_llm_messages_preserve_visible_content_and_tool_calls(
    span: Span,
    #[case] prompt: &str,
) {
    let completion = r#"{
        "generations": [[{"message": {"kwargs": {
            "type": "ai",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "thinking", "thinking": "hidden"},
                {"type": "tool_use", "id": "call-1"},
                {"type": "text", "text": "second"}
            ],
            "tool_calls": [{"name": "search", "args": {"query": "hello"}, "id": "call-1"}],
            "response_metadata": {"id": "response-1"}
        }}}]]
    }"#;
    let decoded = decode_normalization(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "llm"),
            ("gen_ai.prompt", prompt),
            ("gen_ai.completion", completion),
        ],
    )
    .unwrap();
    let input: serde_json::Value = serde_json::from_str(&decoded.normalized.input).unwrap();
    let output: serde_json::Value = serde_json::from_str(&decoded.normalized.output).unwrap();
    assert_eq!(
        input,
        serde_json::json!([{"role": "user", "content": "hello"}])
    );
    assert_eq!(
        output,
        serde_json::json!({
            "role": "assistant",
            "content": "first\n\nsecond",
            "tool_calls": [{"name": "search", "args": {"query": "hello"}, "id": "call-1"}],
        })
    );
    assert_eq!(decoded.normalized.litellm_request_id, "response-1");
}

#[rstest]
#[case::string(r#""result""#, "result")]
#[case::wrapped(r#"{"output":{"content":"result"}}"#, "result")]
#[case::command(
    r#"{"output":{"update":{"messages":[{"content":"ignored"},{"content":"result"}]}}}"#,
    "result"
)]
#[case::object(r#"{"output":{"count":2}}"#, r#"{"count": 2}"#)]
#[case::null(r#"{"output":null}"#, "null")]
fn langsmith_tool_output_unwraps_supported_shapes(
    span: Span,
    #[case] completion: &str,
    #[case] expected: &str,
) {
    let decoded = decode_normalization(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "tool"),
            ("gen_ai.prompt", "raw-tool-input"),
            ("gen_ai.completion", completion),
        ],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Tool);
    assert_eq!(decoded.normalized.input, "raw-tool-input");
    assert_eq!(decoded.normalized.output, expected);
}
