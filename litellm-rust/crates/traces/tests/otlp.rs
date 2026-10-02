use litellm_traces::decode_otlp;
use litellm_traces::{ObservationType, Shared};
use rstest::rstest;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"
);

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
        Err(litellm_traces::DecodeError::InvalidPayload)
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
        Err(litellm_traces::DecodeError::TooLarge)
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
        Err(litellm_traces::DecodeError::TooLarge)
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
        Err(litellm_traces::DecodeError::TooLarge)
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
        Err(litellm_traces::DecodeError::TooLarge)
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

const CLAUDE_AGENT_SDK_FIXTURE: &[u8] =
    include_bytes!("../../../../tests/test_litellm/tracing/fixtures/claude_agent_sdk_export.json");
const CLAUDE_AGENT_SDK_DETAILED_FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/claude_agent_sdk_detailed_export.json"
);

fn raw_spans(fixture: &[u8]) -> Vec<serde_json::Value> {
    let export: serde_json::Value = serde_json::from_slice(fixture).expect("fixture JSON");
    export["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .expect("spans")
        .clone()
}

fn raw_attribute(span: &serde_json::Value, key: &str) -> Option<serde_json::Value> {
    span["attributes"]
        .as_array()
        .expect("attributes")
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| attribute["value"].clone())
}

fn raw_string(span: &serde_json::Value, key: &str) -> String {
    raw_attribute(span, key)
        .and_then(|value| value["stringValue"].as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn raw_int(span: &serde_json::Value, key: &str) -> u64 {
    raw_attribute(span, key).map_or(0, |value| match &value["intValue"] {
        serde_json::Value::String(text) => text.parse().expect("integer"),
        number => number.as_u64().expect("integer"),
    })
}

fn raw_span<'a>(raw: &'a [serde_json::Value], span_id: &str) -> &'a serde_json::Value {
    raw.iter()
        .find(|span| {
            span["spanId"]
                .as_str()
                .is_some_and(|id| id.eq_ignore_ascii_case(span_id))
        })
        .expect("raw span")
}

#[rstest]
#[case::default_telemetry(CLAUDE_AGENT_SDK_FIXTURE)]
#[case::detailed_telemetry(CLAUDE_AGENT_SDK_DETAILED_FIXTURE)]
fn normalizes_claude_agent_sdk_fixture(#[case] fixture: &[u8]) {
    let spans = decode_otlp(fixture, Some("application/json")).expect("valid OTLP export");
    let raw = raw_spans(fixture);
    let types: std::collections::BTreeSet<_> = spans
        .iter()
        .map(|span| format!("{:?}", span.normalized.observation_type))
        .collect();
    assert_eq!(
        types,
        ["Agent", "Framework", "Llm", "Tool"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    );

    let root = spans
        .iter()
        .find(|span| span.normalized.observation_type == ObservationType::Agent)
        .expect("interaction root");
    assert!(root.parent_span_id.is_empty());
    let root_input: serde_json::Value =
        serde_json::from_str(&root.normalized.input).expect("root input messages");
    assert_eq!(root_input[0]["role"], "user");
    assert_eq!(
        root_input[0]["content"],
        raw_string(raw_span(&raw, &root.span_id), "user_prompt")
    );
    assert!(root.consumed_attributes.contains(&"user_prompt"));

    let tools: Vec<_> = spans
        .iter()
        .filter(|span| span.normalized.observation_type == ObservationType::Tool)
        .collect();
    assert_eq!(tools.len(), 2);
    for tool in &tools {
        assert_eq!(
            tool.name,
            raw_string(raw_span(&raw, &tool.span_id), "tool_name")
        );
        let input: serde_json::Value =
            serde_json::from_str(&tool.normalized.input).expect("tool argument object");
        assert!(input.is_object());
        assert!(input.get("role").is_none());
        let event = tool
            .events
            .iter()
            .find(|event| event.name == "tool.output")
            .expect("tool output event");
        let expected_output = ["output", "content", "diff"]
            .into_iter()
            .filter_map(|key| event.attributes.get(key))
            .find(|value| !value.is_empty())
            .expect("event output");
        assert_eq!(&tool.normalized.output, expected_output);
    }
    let bash = tools
        .iter()
        .find(|tool| tool.name == "Bash")
        .expect("Bash tool");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&bash.normalized.input).unwrap()["command"],
        raw_string(raw_span(&raw, &bash.span_id), "full_command")
    );

    let llms: Vec<_> = spans
        .iter()
        .filter(|span| span.normalized.observation_type == ObservationType::Llm)
        .collect();
    assert!(!llms.is_empty());
    for llm in &llms {
        let raw_llm = raw_span(&raw, &llm.span_id);
        let expected = raw_int(raw_llm, "input_tokens")
            + raw_int(raw_llm, "cache_read_tokens")
            + raw_int(raw_llm, "cache_creation_tokens");
        assert_eq!(u64::from(llm.normalized.input_tokens), expected);
        assert_eq!(
            u64::from(llm.normalized.output_tokens),
            raw_int(raw_llm, "output_tokens")
        );
        assert_eq!(llm.normalized.model, raw_string(raw_llm, "model"));
        if raw_string(raw_llm, "query_source_safe") == "sdk" {
            assert_eq!(llm.normalized.framework, "claude-agent-sdk");
        }
    }
    assert!(spans.iter().all(|span| {
        span.normalized.agent_name == span.resource_attributes["service.name"].as_str()
    }));
}

#[rstest]
fn claude_agent_sdk_detailed_fixture_keeps_full_tool_arguments_and_llm_messages() {
    let spans = decode_otlp(CLAUDE_AGENT_SDK_DETAILED_FIXTURE, Some("application/json"))
        .expect("valid OTLP export");
    let raw = raw_spans(CLAUDE_AGENT_SDK_DETAILED_FIXTURE);
    let bash = spans
        .iter()
        .find(|span| span.name == "Bash")
        .expect("Bash tool");
    let tool_input = raw_string(raw_span(&raw, &bash.span_id), "tool_input");
    let (_, arguments) = tool_input.split_once('\n').expect("tool input header");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&bash.normalized.input).unwrap(),
        serde_json::from_str::<serde_json::Value>(arguments).unwrap()
    );
    assert!(bash.consumed_attributes.contains(&"tool_input"));

    let answer = spans
        .iter()
        .find(|span| {
            span.normalized.observation_type == ObservationType::Llm
                && span.attributes.get("query_source_safe").map(String::as_str) == Some("sdk")
                && !span.normalized.output.is_empty()
        })
        .expect("final SDK answer");
    let raw_answer = raw_span(&raw, &answer.span_id);
    let input: serde_json::Value =
        serde_json::from_str(&answer.normalized.input).expect("llm input messages");
    assert_eq!(input[0]["role"], "system");
    assert_eq!(
        input[0]["content"],
        raw_string(raw_answer, "system_prompt_preview")
    );
    let output: serde_json::Value =
        serde_json::from_str(&answer.normalized.output).expect("llm output message");
    assert_eq!(output["role"], "assistant");
    assert_eq!(
        output["content"],
        raw_string(raw_answer, "response.model_output")
    );

    let title = spans
        .iter()
        .find(|span| {
            span.attributes.get("query_source_safe").map(String::as_str)
                == Some("generate_session_title")
        })
        .expect("side query");
    assert_eq!(title.normalized.framework, "claude-agent-sdk");
}

#[rstest]
fn claude_code_scope_takes_precedence_over_openinference_attributes(
    mut span: opentelemetry_proto::tonic::trace::v1::Span,
) {
    use opentelemetry_proto::tonic::common::v1::{
        AnyValue, InstrumentationScope, KeyValue, any_value::Value,
    };
    let string = |key: &str, value: &str| KeyValue {
        key: key.to_owned(),
        value: Some(AnyValue {
            value: Some(Value::StringValue(value.to_owned())),
        }),
        ..Default::default()
    };
    span.attributes = vec![
        string("span.type", "tool"),
        string("tool_name", "Grep"),
        string("openinference.span.kind", "LLM"),
    ];
    let mut request = request_with(span);
    request.resource_spans[0].scope_spans[0].scope = Some(InstrumentationScope {
        name: "com.anthropic.claude_code.tracing".to_owned(),
        ..Default::default()
    });
    let spans = decode_otlp(&prost::Message::encode_to_vec(&request), None).expect("valid span");
    assert_eq!(spans[0].normalized.observation_type, ObservationType::Tool);
    assert_eq!(spans[0].name, "Grep");
    assert_eq!(spans[0].normalized.framework, "claude-code");
    assert_eq!(spans[0].normalized.agent_name, "claude-code");
}
