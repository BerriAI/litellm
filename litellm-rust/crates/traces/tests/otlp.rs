use litellm_traces::decode_otlp;
use litellm_traces::{AgentType, Integration, ObservationType, Shared};
use opentelemetry_proto::tonic::trace::v1::Span;
use rstest::rstest;

const FIXTURE: &[u8] = include_bytes!("fixtures/langsmith_deep_agent_export.json");

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
    assert_eq!(
        llm.normalized.agent_name.as_deref().unwrap_or_default(),
        "deep_research_agent"
    );
    assert_eq!(
        llm.normalized.model.as_deref().unwrap_or_default(),
        "claude-sonnet-4-5"
    );
    assert_eq!(
        (llm.normalized.input_tokens, llm.normalized.output_tokens),
        (3332, 467)
    );
    assert!(llm.normalized.calls.key_set().unwrap().contains(
        &litellm_traces::CallKey::ProviderResponse(
            "chatcmpl-4077bb36-9380-4a3b-9481-245700cef09a".to_owned()
        )
    ));
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
    decode_normalization_with_resources(span, scope, attributes, &[])
}

fn decode_normalization_with_resources(
    span: Span,
    scope: &str,
    attributes: &[(&str, &str)],
    resources: &[(&str, &str)],
) -> Result<litellm_traces::DecodedSpan, litellm_traces::Error> {
    use opentelemetry_proto::tonic::{
        collector::trace::v1::ExportTraceServiceRequest,
        common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value::Value},
        resource::v1::Resource,
        trace::v1::{ResourceSpans, ScopeSpans},
    };
    use prost::Message;

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: resources
                    .iter()
                    .map(|(key, value)| KeyValue {
                        key: (*key).to_owned(),
                        value: Some(AnyValue {
                            value: Some(Value::StringValue((*value).to_owned())),
                        }),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
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
#[case::embedding("embeddings", true, ObservationType::Embedding)]
#[case::retrieval("retrieval", false, ObservationType::Retriever)]
#[case::workflow("invoke_workflow", true, ObservationType::Chain)]
#[case::create_agent("create_agent", false, ObservationType::Framework)]
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
#[case::retriever("RETRIEVER", ObservationType::Retriever)]
#[case::embedding("EMBEDDING", ObservationType::Embedding)]
#[case::reranker("RERANKER", ObservationType::Reranker)]
#[case::guardrail("GUARDRAIL", ObservationType::Guardrail)]
#[case::evaluator("EVALUATOR", ObservationType::Evaluator)]
#[case::prompt("PROMPT", ObservationType::Prompt)]
#[case::decision("DECISION", ObservationType::Decision)]
fn openinference_preserves_operation_and_payload_at_any_depth(
    span: Span,
    #[case] kind: &str,
    #[case] expected: ObservationType,
    #[values(true, false)] root: bool,
) {
    let input = r#"{"query":"hello"}"#;
    let output = r#"[{"id":"doc-1","score":0.9}]"#;
    let decoded = decode_normalization(
        Span {
            parent_span_id: if root { vec![] } else { vec![3; 8] },
            ..span
        },
        "openinference.instrumentation.example",
        &[
            ("openinference.span.kind", kind),
            ("input.value", input),
            ("output.value", output),
        ],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
    assert!(!decoded.normalized.wrapper_candidate);
    assert_eq!(decoded.normalized.input, input);
    assert_eq!(decoded.normalized.output, output);
    assert_eq!(
        serde_json::to_value(decoded.normalized.calls.kind()).unwrap(),
        "unknown"
    );
}

#[rstest]
#[case::claude("claude-code", Integration::ClaudeCode)]
#[case::codex("openai-codex", Integration::OpenaiCodex)]
#[case::deepagents("deepagents-code", Integration::DeepagentsCode)]
#[case::cursor("cursor", Integration::Cursor)]
#[case::pi("pi", Integration::Pi)]
#[case::opencode("opencode", Integration::Opencode)]
#[case::copilot("copilot", Integration::Copilot)]
#[case::extension("future-agent", Integration::Other("future-agent".to_owned()))]
fn coding_identity_is_independent_of_model_operation(
    span: Span,
    #[case] integration: &str,
    #[case] expected: Integration,
) {
    let decoded = decode_normalization(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "llm"),
            ("langsmith.metadata.ls_agent_type", "subagent"),
            ("langsmith.metadata.ls_integration", integration),
            ("langsmith.metadata.thread_id", "thread-1"),
            ("langsmith.metadata.ls_subagent_id", "agent-1"),
            ("langsmith.metadata.ls_subagent_type", "researcher"),
            ("langsmith.metadata.ls_model_name", "test-model"),
        ],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, ObservationType::Llm);
    assert_eq!(
        decoded
            .normalized
            .framework
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        integration
    );
    assert_eq!(
        decoded.normalized.model.as_deref().unwrap_or_default(),
        "test-model"
    );
    assert_eq!(
        decoded.normalized.agent_name.as_deref().unwrap_or_default(),
        "researcher"
    );
    assert_eq!(
        serde_json::to_value(decoded.normalized.calls.kind()).unwrap(),
        "unknown"
    );
    assert!(
        decoded
            .normalized
            .calls
            .key_set()
            .is_none_or(|keys| keys.is_empty())
    );
    let metadata = &decoded.normalized.agent_metadata;
    assert_eq!(metadata.ls_integration, Some(expected));
    assert_eq!(metadata.ls_agent_type, Some(AgentType::Subagent));
    assert_eq!(metadata.thread_id.as_deref(), Some("thread-1"));
    assert_eq!(metadata.ls_subagent_id.as_deref(), Some("agent-1"));
    assert_eq!(
        serde_json::to_value(metadata).unwrap()["ls_integration"],
        integration
    );
}

#[rstest]
#[case::subagent("subagent", "chain", ObservationType::Agent)]
#[case::root("root", "chain", ObservationType::Agent)]
#[case::middleware("middleware", "chain", ObservationType::Framework)]
#[case::compaction("compaction", "chain", ObservationType::Framework)]
#[case::compaction_model("compaction", "llm", ObservationType::Llm)]
#[case::middleware_tool("middleware", "tool", ObservationType::Tool)]
#[case::retrieval("root", "retriever", ObservationType::Retriever)]
fn agent_context_only_refines_container_roles(
    span: Span,
    #[case] agent_type: &str,
    #[case] kind: &str,
    #[case] expected: ObservationType,
) {
    let decoded = decode_normalization(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", kind),
            ("langsmith.metadata.ls_agent_type", agent_type),
        ],
    )
    .unwrap();
    assert_eq!(decoded.normalized.observation_type, expected);
    assert!(!decoded.normalized.wrapper_candidate);
}

#[rstest]
fn metadata_sources_merge_with_flattened_values_taking_precedence(span: Span) {
    let decoded = decode_normalization(
        span,
        "langsmith",
        &[
            ("langsmith.span.kind", "tool"),
            ("metadata", r#"{"ls_integration":"cursor","thread_id":"nested","ls_agent_type":42,"ls_agent_runtime":"runtime","ls_provider":"test-provider","repository_url":"repo","cwd":"directory","ls_agent_runtime_version":"version"}"#),
            ("thread_id", "direct"),
            ("langsmith.metadata.thread_id", "flattened"),
            ("langsmith.metadata.ls_tool_name", "shell"),
            ("langsmith.metadata.ls_agent_type", "unknown-context"),
        ],
    ).unwrap();
    let metadata = &decoded.normalized.agent_metadata;
    assert_eq!(metadata.thread_id.as_deref(), Some("flattened"));
    assert_eq!(metadata.ls_agent_type, None);
    assert_eq!(metadata.ls_agent_runtime.as_deref(), Some("runtime"));
    assert_eq!(metadata.ls_provider.as_deref(), Some("test-provider"));
    assert_eq!(metadata.git_repo_url.as_deref(), Some("repo"));
    assert_eq!(metadata.working_directory.as_deref(), Some("directory"));
    assert_eq!(metadata.ls_agent_version.as_deref(), Some("version"));
    assert_eq!(decoded.name, "shell");
    assert_eq!(decoded.normalized.observation_type, ObservationType::Tool);
    assert_eq!(
        decoded
            .normalized
            .framework
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        "cursor"
    );
}

#[rstest]
fn metadata_projection_respects_the_decoded_byte_budget(span: Span) {
    let thread = "x".repeat(9 * 1024 * 1024);
    assert!(matches!(
        decode_normalization(span, "example", &[("thread_id", &thread)]),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
fn genai_retrieval_normalizes_query_and_documents(span: Span) {
    let query = "trace storage";
    let documents = r#"[{"id":"doc-1","score":0.9}]"#;
    let decoded = decode_normalization(
        span,
        "example",
        &[
            ("gen_ai.operation.name", "retrieval"),
            ("gen_ai.retrieval.query.text", query),
            ("gen_ai.retrieval.documents", documents),
        ],
    )
    .unwrap();
    assert_eq!(
        decoded.normalized.observation_type,
        ObservationType::Retriever
    );
    assert_eq!(decoded.normalized.input, query);
    assert_eq!(decoded.normalized.output, documents);
    assert_eq!(decoded.normalized.input_preview, query);
}

#[rstest]
#[case::image(serde_json::json!({"type": "image_url", "image_url": {"url": "image"}}))]
#[case::unknown(serde_json::json!({"type": "unknown", "payload": "opaque"}))]
#[case::malformed(serde_json::json!({"type": "text", "text": 7}))]
#[case::scalar(serde_json::json!(7))]
fn genai_message_blocks_preserve_text_without_exposing_hidden_content(
    span: Span,
    #[case] unsupported: serde_json::Value,
    #[values(
        "reasoning",
        "thinking",
        "redacted_thinking",
        "function_call",
        "tool_use",
        "tool_call"
    )]
    hidden_type: &str,
) {
    let payload = serde_json::json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "first"},
            {"type": hidden_type, "text": "hidden", "thinking": "hidden", "input": "hidden"},
            unsupported,
            {"text": "second"},
        ],
    }])
    .to_string();
    let decoded = decode_normalization(
        span,
        "",
        &[
            ("gen_ai.input.messages", &payload),
            ("gen_ai.output.messages", &payload),
        ],
    )
    .unwrap();
    let expected = serde_json::json!([{"role": "user", "content": "first\n\nsecond"}]);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&decoded.normalized.input).unwrap(),
        expected,
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&decoded.normalized.output).unwrap(),
        expected,
    );
}

#[rstest]
#[case::text(serde_json::json!("hello"), "hello")]
#[case::object(serde_json::json!({"count": 2}), r#"{"count": 2}"#)]
#[case::number(serde_json::json!(7), "7")]
#[case::empty_blocks(serde_json::json!([]), "")]
fn genai_message_content_preserves_text_and_non_array_fallbacks(
    span: Span,
    #[case] content: serde_json::Value,
    #[case] expected: &str,
) {
    let payload = serde_json::json!([{"role": "user", "content": content}]).to_string();
    let decoded = decode_normalization(span, "", &[("gen_ai.input.messages", &payload)]).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&decoded.normalized.input).unwrap(),
        serde_json::json!([{"role": "user", "content": expected}]),
    );
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
            fields.model.as_deref().unwrap_or_default(),
            fields.input.as_str(),
            fields.output.as_str()
        ],
        expected
    );
    assert_eq!(fields.agent_name.as_deref(), Some("test-agent"));
    assert!(
        fields
            .calls
            .key_set()
            .unwrap()
            .contains(&litellm_traces::CallKey::ProviderResponse(
                "response-1".to_owned()
            ))
    );
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
    assert_eq!(fields.model.as_deref(), Some("inference-model"));
    assert_eq!(fields.agent_name.as_deref(), Some("inference-agent"));
    assert_eq!(fields.input, "inference-input");
    assert_eq!(fields.output, "inference-output");
    assert_eq!(
        (fields.input_tokens, fields.output_tokens),
        (input_tokens, output_tokens)
    );
    assert_eq!(
        *decoded.consumed_attributes,
        ["input.value", "output.value"]
    );
}

#[rstest]
#[case::raw_response("LLM", r#"{"id":"chatcmpl-1","choices":[]}"#, &["provider_response:chatcmpl-1"], "complete")]
#[case::wrapped_response("LLM", r#"{"raw":{"id":"wrapped"}}"#, &["provider_response:wrapped"], "complete")]
#[case::top_level_wins("LLM", r#"{"id":"direct","raw":{"id":"wrapped"}}"#, &["provider_response:direct"], "complete")]
#[case::null_top_level_shadows_raw("LLM", r#"{"id":null,"raw":{"id":"wrapped"}}"#, &[], "unknown")]
#[case::invalid_top_level_shadows_raw("LLM", r#"{"id":7,"raw":{"id":"wrapped"}}"#, &[], "unknown")]
#[case::array_raw_is_not_a_response("LLM", r#"{"raw":["wrapped"]}"#, &[], "unknown")]
#[case::invalid_raw_keeps_top_level("LLM", r#"{"id":"direct","raw":7}"#, &["provider_response:direct"], "complete")]
#[case::langchain_llm_output("LLM", r#"{"llm_output":{"id":"chatcmpl-2"},"generations":[[{"message":{"kwargs":{"type":"ai","content":"hi"}}}]]}"#, &["provider_response:chatcmpl-2"], "complete")]
#[case::langchain_generation("LLM", r#"{"generations":[[{"message":{"kwargs":{"response_metadata":{"id":"chatcmpl-3"}}}}]]}"#, &["provider_response:chatcmpl-3"], "complete")]
#[case::langchain_batch("LLM", r#"{"generations":[[{"message":{"kwargs":{"response_metadata":{"id":"a"}}}}],[{"message":{"kwargs":{}}}]]}"#, &["provider_response:a"], "partial")]
#[case::malformed_candidate("LLM", r#"{"generations":[[{"message":{"kwargs":{"response_metadata":{"id":"a"}}}},null]]}"#, &["provider_response:a"], "partial")]
#[case::malformed_prompt("LLM", r#"{"generations":[null,[{"message":{"response_metadata":{"id":"a"}}}]]}"#, &["provider_response:a"], "partial")]
#[case::invalid_candidate_id("LLM", r#"{"generations":[[{"message":{"response_metadata":{"id":"a"}}},{"message":{"response_metadata":{"id":7}}}]]}"#, &["provider_response:a"], "partial")]
#[case::shared_candidate_id("LLM", r#"{"generations":[[{"message":{"response_metadata":{"id":"a"}}},{"message":{"response_metadata":{"id":"a"}}}]]}"#, &["provider_response:a"], "complete")]
#[case::conflicting_candidate_ids("LLM", r#"{"generations":[[{"message":{"response_metadata":{"id":"a"}}},{"message":{"response_metadata":{"id":"b"}}}]]}"#, &["provider_response:a", "provider_response:b"], "partial")]
#[case::multiple_prompt_ids("LLM", r#"{"generations":[[{"message":{"response_metadata":{"id":"a"}}}],[{"message":{"response_metadata":{"id":"b"}}}]]}"#, &["provider_response:a", "provider_response:b"], "complete")]
#[case::fallback_with_invalid_candidate("LLM", r#"{"llm_output":{"id":"a"},"generations":[[null]]}"#, &["provider_response:a"], "partial")]
#[case::empty_generations("LLM", r#"{"llm_output":{"id":"a"},"generations":[]}"#, &[], "unknown")]
#[case::non_llm("CHAIN", r#"{"id":"task-1"}"#, &[], "unknown")]
#[case::not_json("LLM", "plain text", &[], "unknown")]
#[case::non_string_id("LLM", r#"{"id":7}"#, &[], "unknown")]
fn openinference_llm_output_records_call_evidence(
    span: Span,
    #[case] kind: &str,
    #[case] output: &str,
    #[case] keys: &[&str],
    #[case] evidence: &str,
) {
    let decoded = decode_normalization(
        span,
        "",
        &[("openinference.span.kind", kind), ("output.value", output)],
    )
    .unwrap();
    let recorded: Vec<String> = decoded
        .normalized
        .calls
        .key_set()
        .into_iter()
        .flatten()
        .map(ToString::to_string)
        .collect();
    assert_eq!(recorded, keys);
    assert_eq!(
        serde_json::to_value(decoded.normalized.calls.kind()).unwrap(),
        evidence
    );
}

#[rstest]
#[case::crewai("openinference.instrumentation.crewai", "crewai")]
#[case::multi_word("openinference.instrumentation.claude_agent_sdk", "claude-agent-sdk")]
#[case::other_scope("other", "")]
fn openinference_scope_names_the_framework(
    span: Span,
    #[case] scope: &str,
    #[case] framework: &str,
) {
    let decoded =
        decode_normalization(span, scope, &[("openinference.span.kind", "AGENT")]).unwrap();
    assert_eq!(
        decoded
            .normalized
            .framework
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        framework
    );
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
    assert_eq!(
        decoded.normalized.agent_name.as_deref().unwrap_or_default(),
        "test-agent"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&decoded.normalized.input).unwrap(),
        serde_json::json!([{"role": "user", "content": "hello"}]),
    );
    assert_eq!(
        *decoded.consumed_attributes,
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
                {"type": "image_url", "image_url": {"url": "image"}},
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
    assert!(decoded.normalized.calls.key_set().unwrap().contains(
        &litellm_traces::CallKey::ProviderResponse("response-1".to_owned())
    ));
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

const CLAUDE_AGENT_SDK_FIXTURE: &[u8] = include_bytes!("fixtures/claude_agent_sdk_export.json");
const CLAUDE_AGENT_SDK_DETAILED_FIXTURE: &[u8] =
    include_bytes!("fixtures/claude_agent_sdk_detailed_export.json");

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
        assert_eq!(
            llm.normalized.model.as_deref().unwrap_or_default(),
            raw_string(raw_llm, "model")
        );
        if raw_string(raw_llm, "query_source_safe") == "sdk" {
            assert_eq!(
                llm.normalized
                    .framework
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
                "claude-agent-sdk"
            );
        }
    }
    assert!(spans.iter().all(|span| {
        span.normalized.agent_name.as_deref().unwrap_or_default()
            == span.resource_attributes["service.name"].as_str()
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
    assert_eq!(
        title
            .normalized
            .framework
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        "claude-agent-sdk"
    );
}

#[rstest]
fn claude_code_scope_takes_precedence_over_other_conventions(span: Span) {
    let decoded = decode_normalization(
        span,
        "com.anthropic.claude_code.tracing",
        &[
            ("span.type", "tool"),
            ("tool_name", "Grep"),
            ("openinference.span.kind", "LLM"),
            ("langsmith.span.kind", "LLM"),
        ],
    )
    .expect("valid span");
    assert_eq!(decoded.normalized.observation_type, ObservationType::Tool);
    assert_eq!(decoded.name, "Grep");
    assert_eq!(
        decoded
            .normalized
            .framework
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
        "claude-code"
    );
    assert_eq!(
        decoded.normalized.agent_name.as_deref().unwrap_or_default(),
        "claude-code"
    );
}

#[rstest]
#[case::sdk_wrapper("openinference.instrumentation.claude_agent_sdk", &[("openinference.span.kind", "AGENT"), ("agent.name", "Agent")], &[("gen_ai.agent.name", "worker")], "worker")]
#[case::generic_fallback("custom", &[], &[("gen_ai.agent.name", "worker")], "worker")]
#[case::generic_explicit("custom", &[("gen_ai.agent.name", "explicit")], &[("gen_ai.agent.name", "worker")], "explicit")]
#[case::generic_service("custom", &[], &[("service.name", "worker")], "")]
#[case::hermes_default("hermes-otel-plugin", &[("gen_ai.agent.name", "hermes-agent")], &[("gen_ai.agent.name", "worker")], "worker")]
#[case::hermes_explicit("hermes-otel-plugin", &[("gen_ai.agent.name", "explicit")], &[("gen_ai.agent.name", "worker")], "explicit")]
#[case::claude_default("com.anthropic.claude_code.tracing", &[], &[("gen_ai.agent.name", "worker"), ("service.name", "service")], "worker")]
#[case::claude_service("com.anthropic.claude_code.tracing", &[], &[("service.name", "service")], "service")]
#[case::claude_empty_resource_name("com.anthropic.claude_code.tracing", &[], &[("gen_ai.agent.name", ""), ("service.name", "service")], "service")]
#[case::claude_subagent("com.anthropic.claude_code.tracing", &[("span.type", "llm_request"), ("query_source", "agent:custom:delegate")], &[("gen_ai.agent.name", "worker"), ("service.name", "service")], "delegate")]
fn resource_identity_preserves_explicit_names_and_sdk_fallbacks(
    span: Span,
    #[case] scope: &str,
    #[case] attributes: &[(&str, &str)],
    #[case] resources: &[(&str, &str)],
    #[case] expected: &str,
) {
    let decoded = decode_normalization_with_resources(span, scope, attributes, resources)
        .expect("valid span");
    assert_eq!(
        decoded.normalized.agent_name.as_deref().unwrap_or_default(),
        expected
    );
}

#[rstest]
#[case::depth(litellm_traces::DecodeLimits { depth: 1, ..Default::default() })]
#[case::nodes(litellm_traces::DecodeLimits { nodes: 1, ..Default::default() })]
#[case::spans(litellm_traces::DecodeLimits { spans: 1, ..Default::default() })]
#[case::attributes(litellm_traces::DecodeLimits { attributes: 1, ..Default::default() })]
#[case::events(litellm_traces::DecodeLimits { events: 1, ..Default::default() })]
#[case::links(litellm_traces::DecodeLimits { links: 1, ..Default::default() })]
#[case::decoded_bytes(litellm_traces::DecodeLimits { decoded_span_bytes: 1, ..Default::default() })]
fn configurable_decode_limits_apply_to_both_wire_formats(
    mut span: Span,
    #[case] limits: litellm_traces::DecodeLimits,
) {
    use opentelemetry_proto::tonic::{
        common::v1::KeyValue,
        trace::v1::span::{Event, Link},
    };
    use prost::Message;
    span.attributes = vec![
        KeyValue {
            key: "a".into(),
            ..Default::default()
        },
        KeyValue {
            key: "b".into(),
            ..Default::default()
        },
    ];
    span.events = vec![Event::default(), Event::default()];
    span.links = vec![
        Link {
            trace_id: vec![1; 16],
            span_id: vec![2; 8],
            ..Default::default()
        };
        2
    ];
    let mut request = request_with(span.clone());
    request.resource_spans[0].scope_spans[0].spans.push(span);
    for (body, content_type) in [
        (serde_json::to_vec(&request).unwrap(), "application/json"),
        (request.encode_to_vec(), "application/x-protobuf"),
    ] {
        assert!(matches!(
            litellm_traces::decode_otlp_with_limits(&body, Some(content_type), limits),
            Err(litellm_traces::Error::TooLarge)
        ));
        assert_eq!(
            litellm_traces::decode_otlp_with_limits(
                &body,
                Some(content_type),
                litellm_traces::DecodeLimits::default()
            )
            .unwrap()
            .len(),
            2
        );
    }
}

#[test]
fn environment_decode_limits_are_used_and_invalid_values_fail() {
    for value in ["2", "4", "0", "invalid"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "environment_decode_limits_child"])
            .env("LITELLM_TEST_DECODE_LIMIT", value)
            .env("OTLP_MAX_SPANS", value)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
    }
}

#[test]
fn environment_decode_limits_child() {
    let Ok(value) = std::env::var("LITELLM_TEST_DECODE_LIMIT") else {
        return;
    };
    let result = decode_otlp(
        include_bytes!("fixtures/opentelemetry_simple.json"),
        Some("application/json"),
    );
    match value.as_str() {
        "2" => assert!(matches!(result, Err(litellm_traces::Error::TooLarge))),
        "4" => assert_eq!(result.unwrap().len(), 3),
        _ => assert!(matches!(
            result,
            Err(litellm_traces::Error::InvalidLimit("OTLP_MAX_SPANS"))
        )),
    }
}

fn log_request(
    source: &str,
) -> opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest {
    use opentelemetry_proto::tonic::{
        collector::logs::v1::ExportLogsServiceRequest,
        common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value::Value},
        logs::v1::{LogRecord, ResourceLogs, ScopeLogs},
    };
    let attributes = [
        ("event.name", "assistant_response"),
        ("query_source", source),
        ("response", "Visible reply"),
        ("message.uuid", "message-one"),
        ("model", "test-model"),
        ("session.id", "session-one"),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.to_owned(),
        value: Some(AnyValue {
            value: Some(Value::StringValue(value.to_owned())),
        }),
        ..Default::default()
    })
    .collect();
    ExportLogsServiceRequest {
        resource_logs: vec![ResourceLogs {
            scope_logs: vec![ScopeLogs {
                scope: Some(InstrumentationScope {
                    name: "com.anthropic.claude_code.events".to_owned(),
                    ..Default::default()
                }),
                log_records: vec![LogRecord {
                    trace_id: vec![1; 16],
                    span_id: vec![2; 8],
                    time_unix_nano: 100,
                    attributes,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

#[rstest]
#[case::main("repl_main_thread", 1)]
#[case::subagent("agent:builtin:general-purpose", 1)]
#[case::title("generate_session_title", 0)]
#[case::suggestion("prompt_suggestion", 0)]
fn native_assistant_logs_preserve_visible_messages_without_counting_model_calls(
    #[case] source: &str,
    #[case] count: usize,
) {
    use prost::Message;
    let request = log_request(source);
    let json = litellm_traces::decode_otlp_logs(
        &serde_json::to_vec(&request).unwrap(),
        Some("application/json"),
    )
    .unwrap();
    let binary = litellm_traces::decode_otlp_logs(&request.encode_to_vec(), None).unwrap();
    assert_eq!(
        serde_json::to_value(&json).unwrap(),
        serde_json::to_value(&binary).unwrap()
    );
    assert_eq!(json.len(), count);
    if let Some(span) = json.first() {
        assert_eq!(span.trace_id, "01".repeat(16));
        assert_eq!(span.parent_span_id, "02".repeat(8));
        assert_ne!(span.span_id, span.parent_span_id);
        assert_eq!(span.normalized.observation_type, ObservationType::Chain);
        assert_eq!(span.normalized.framework, Some(Integration::ClaudeCode));
        assert_eq!(span.normalized.model.as_deref(), Some("test-model"));
        assert_eq!(span.normalized.output_tokens, 0);
        assert_eq!(span.normalized.input_tokens, 0);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&span.normalized.output).unwrap()["content"],
            "Visible reply"
        );
    }
}

#[rstest]
#[case::json(true)]
#[case::protobuf(false)]
fn simultaneous_native_tool_logs_keep_distinct_sequence_ids(#[case] json: bool) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    let template = request.resource_logs[0].scope_logs[0].log_records[0].clone();
    request.resource_logs[0].scope_logs[0].log_records = [1, 2]
        .into_iter()
        .map(|sequence| {
            let mut record = template.clone();
            record.attributes = [
                ("event.name", Value::StringValue("tool_result".into())),
                ("event.sequence", Value::IntValue(sequence)),
                (
                    "tool_use_id",
                    Value::StringValue(format!("call-{sequence}")),
                ),
            ]
            .into_iter()
            .map(|(key, value)| KeyValue {
                key: key.into(),
                value: Some(AnyValue { value: Some(value) }),
                ..Default::default()
            })
            .collect();
            record
        })
        .collect();
    let bytes = if json {
        serde_json::to_vec(&request).unwrap()
    } else {
        request.encode_to_vec()
    };
    let content_type = json.then_some("application/json");
    let spans = litellm_traces::decode_otlp_logs(&bytes, content_type).unwrap();
    assert_eq!(spans.len(), 2);
    assert_ne!(spans[0].span_id, spans[1].span_id);
    let replayed = litellm_traces::decode_otlp_logs(&bytes, content_type).unwrap();
    assert_eq!(spans[0].span_id, replayed[0].span_id);
    assert_eq!(spans[1].span_id, replayed[1].span_id);
}

#[rstest]
#[case::boolean_failure(false, true)]
#[case::boolean_success(true, true)]
#[case::string_failure(false, false)]
#[case::string_success(true, false)]
fn native_tool_log_status_accepts_boolean_and_string_values(
    #[case] success: bool,
    #[case] typed: bool,
) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    request.resource_logs[0].scope_logs[0].log_records[0].attributes = [
        ("event.name", Value::StringValue("tool_result".into())),
        ("error", Value::StringValue("Command failed".into())),
        (
            "success",
            if typed {
                Value::BoolValue(success)
            } else {
                Value::StringValue(success.to_string())
            },
        ),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.into(),
        value: Some(AnyValue { value: Some(value) }),
        ..Default::default()
    })
    .collect();
    let binary = litellm_traces::decode_otlp_logs(&request.encode_to_vec(), None).unwrap();
    let json = litellm_traces::decode_otlp_logs(
        &serde_json::to_vec(&request).unwrap(),
        Some("application/json"),
    )
    .unwrap();
    assert_eq!(binary[0].status_code == "STATUS_CODE_ERROR", !success);
    assert_eq!(json[0].status_code, binary[0].status_code);
    if !success {
        assert_eq!(binary[0].status_message, "Command failed");
    }
}

#[rstest]
fn session_capture_joins_native_logs_and_traces_across_turns_without_changing_span_parents() {
    use opentelemetry_proto::tonic::{
        common::v1::{AnyValue, KeyValue, any_value::Value},
        resource::v1::Resource,
    };
    use prost::Message;
    let mut logs = log_request("repl_main_thread");
    let resource = Resource {
        attributes: [
            ("lens.session.capture", "true"),
            ("gen_ai.agent.name", "custom-claude"),
        ]
        .into_iter()
        .map(|(key, value)| KeyValue {
            key: key.to_owned(),
            value: Some(AnyValue {
                value: Some(Value::StringValue(value.to_owned())),
            }),
            ..Default::default()
        })
        .collect(),
        ..Default::default()
    };
    logs.resource_logs[0].resource = Some(resource.clone());
    let mut request = request_with(Span {
        trace_id: vec![3; 16],
        span_id: vec![4; 8],
        name: "claude_code.interaction".to_owned(),
        attributes: logs.resource_logs[0].scope_logs[0].log_records[0]
            .attributes
            .iter()
            .filter(|attr| attr.key == "session.id")
            .cloned()
            .collect(),
        start_time_unix_nano: 100,
        end_time_unix_nano: 200,
        ..Default::default()
    });
    request.resource_spans[0].resource = Some(resource);
    request.resource_spans[0].scope_spans[0].scope = Some(
        opentelemetry_proto::tonic::common::v1::InstrumentationScope {
            name: "com.anthropic.claude_code.tracing".to_owned(),
            ..Default::default()
        },
    );
    let first = litellm_traces::decode_otlp_logs(&logs.encode_to_vec(), None).unwrap();
    let second = decode_otlp(&request.encode_to_vec(), None).unwrap();
    assert_eq!(first[0].trace_id, second[0].trace_id);
    // Lens feedback resolves session ids the same way; keep in sync with
    // litellm/proxy/lens/feedback_repository.py::session_trace_id.
    assert_eq!(second[0].trace_id, "5fddf060372c8501dca4f331b9da882b");
    assert_eq!(
        first[0].attributes["lens.original_trace_id"],
        "01".repeat(16)
    );
    assert_eq!(
        second[0].attributes["lens.original_trace_id"],
        "03".repeat(16)
    );
    assert_eq!(first[0].parent_span_id, "02".repeat(8));
    assert_eq!(second[0].attributes["gen_ai.agent.id"], "session-one");
    assert_eq!(
        first[0].normalized.agent_name.as_deref(),
        Some("custom-claude")
    );
    request.resource_spans[0].resource = None;
    assert_eq!(
        decode_otlp(&request.encode_to_vec(), None).unwrap()[0].trace_id,
        "03".repeat(16)
    );
}

#[rstest]
#[case::short_trace(vec![1;15], vec![2;8], 1)]
#[case::short_parent(vec![1;16], vec![2;7], 1)]
#[case::timestamp(vec![1;16], vec![2;8], i64::MAX as u64 + 1)]
fn native_logs_reject_invalid_context(
    #[case] trace: Vec<u8>,
    #[case] parent: Vec<u8>,
    #[case] time: u64,
) {
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    let record = &mut request.resource_logs[0].scope_logs[0].log_records[0];
    record.trace_id = trace;
    record.span_id = parent;
    record.time_unix_nano = time;
    assert!(matches!(
        litellm_traces::decode_otlp_logs(&request.encode_to_vec(), None),
        Err(litellm_traces::Error::InvalidPayload)
    ));
}

#[rstest]
#[case::json_empty(true, false, true, true)]
#[case::protobuf_empty(false, false, true, true)]
#[case::json_zero(true, true, true, true)]
#[case::protobuf_zero(false, true, true, true)]
#[case::trace_only(true, false, true, false)]
#[case::parent_only(false, false, false, true)]
fn contextless_native_logs_preserve_the_entire_batch(
    #[case] json: bool,
    #[case] zero: bool,
    #[case] missing_trace: bool,
    #[case] missing_parent: bool,
) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    let valid = request.resource_logs[0].scope_logs[0].log_records[0].clone();
    let mut uncorrelated = valid.clone();
    if missing_trace {
        uncorrelated.trace_id = if zero { vec![0; 16] } else { Vec::new() };
    }
    if missing_parent {
        uncorrelated.span_id = if zero { vec![0; 8] } else { Vec::new() };
    }
    let mut standalone = uncorrelated.clone();
    standalone
        .attributes
        .retain(|attribute| attribute.key != "session.id");
    request.resource_logs[0].scope_logs[0].log_records = vec![valid, uncorrelated, standalone];
    request.resource_logs[0].resource = Some(opentelemetry_proto::tonic::resource::v1::Resource {
        attributes: vec![KeyValue {
            key: "lens.session.capture".into(),
            value: Some(AnyValue {
                value: Some(Value::StringValue("true".into())),
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    let bytes = if json {
        serde_json::to_vec(&request).unwrap()
    } else {
        request.encode_to_vec()
    };
    let media = json.then_some("application/json");
    let limits = litellm_traces::DecodeLimits {
        attributes: 6,
        ..Default::default()
    };
    let spans = litellm_traces::decode_otlp_logs_with_limits(&bytes, media, limits).unwrap();
    let replayed = litellm_traces::decode_otlp_logs_with_limits(&bytes, media, limits).unwrap();
    assert_eq!(spans.len(), 3);
    assert_eq!(
        serde_json::to_value(&spans).unwrap(),
        serde_json::to_value(&replayed).unwrap()
    );
    assert_eq!(spans[0].trace_id, spans[1].trace_id);
    assert_ne!(spans[1].trace_id, spans[2].trace_id);
    assert_ne!(spans[0].span_id, spans[1].span_id);
    if missing_trace {
        assert_ne!(spans[1].span_id, spans[2].span_id);
        assert!(!spans[1].attributes.contains_key("lens.original_trace_id"));
    }
    assert_eq!(spans[0].parent_span_id, "02".repeat(8));
    assert!(spans[1].parent_span_id.is_empty());
    assert!(spans[2].parent_span_id.is_empty());
    assert!(!spans[0].attributes.contains_key("lens.capture.warning"));
    assert!(spans[1].attributes["lens.capture.warning"].contains("unconfirmed"));
    assert!(spans[2].attributes["lens.capture.warning"].contains("unconfirmed"));
    assert_eq!(spans[0].normalized.output, spans[1].normalized.output);
    assert_eq!(spans[0].normalized.output, spans[2].normalized.output);
    assert_eq!(spans[1].normalized.observation_type, ObservationType::Chain);
}

#[rstest]
#[case::session(true)]
#[case::standalone(false)]
fn absent_native_log_context_has_encoding_independent_identity(#[case] session: bool) {
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    let record = &mut request.resource_logs[0].scope_logs[0].log_records[0];
    record.trace_id.clear();
    record.span_id.clear();
    if !session {
        record.attributes.retain(|entry| entry.key != "session.id");
    }
    let omitted = litellm_traces::decode_otlp_logs(
        &serde_json::to_vec(&request).unwrap(),
        Some("application/json"),
    )
    .unwrap();
    let record = &mut request.resource_logs[0].scope_logs[0].log_records[0];
    record.trace_id = vec![0; 16];
    record.span_id = vec![0; 8];
    let zeroed = litellm_traces::decode_otlp_logs(&request.encode_to_vec(), None).unwrap();
    assert_eq!(omitted[0].trace_id, zeroed[0].trace_id);
    assert_eq!(omitted[0].span_id, zeroed[0].span_id);
    assert_eq!(omitted[0].normalized.output, zeroed[0].normalized.output);
}

#[rstest]
#[case::nodes(litellm_traces::DecodeLimits { nodes: 4, ..Default::default() })]
#[case::depth(litellm_traces::DecodeLimits { depth: 2, ..Default::default() })]
#[case::bytes(litellm_traces::DecodeLimits { decoded_span_bytes: 20, ..Default::default() })]
#[case::attributes(litellm_traces::DecodeLimits { attributes: 2, ..Default::default() })]
fn native_logs_enforce_budgets_for_both_encodings(#[case] limits: litellm_traces::DecodeLimits) {
    use prost::Message;
    let request = log_request("repl_main_thread");
    assert!(matches!(
        litellm_traces::decode_otlp_logs_with_limits(&request.encode_to_vec(), None, limits),
        Err(litellm_traces::Error::TooLarge)
    ));
    assert!(matches!(
        litellm_traces::decode_otlp_logs_with_limits(
            &serde_json::to_vec(&request).unwrap(),
            Some("application/json"),
            limits
        ),
        Err(litellm_traces::Error::TooLarge)
    ));
}

#[rstest]
fn interactive_claude_exports_join_replies_with_native_child_execution_context() {
    let traces = decode_otlp(
        include_bytes!("fixtures/claude_code_native_traces.json"),
        Some("application/json"),
    )
    .unwrap();
    let logs = litellm_traces::decode_otlp_logs(
        include_bytes!("fixtures/claude_code_native_logs.json"),
        Some("application/json"),
    )
    .unwrap();
    assert!(
        logs.iter()
            .any(|span| span.normalized.output.contains("MINIMAL-COMMENTARY"))
    );
    assert!(
        logs.iter()
            .any(|span| span.normalized.output.contains("MINIMAL-FINAL"))
    );
    assert!(
        logs.iter()
            .any(|span| span.normalized.output.contains("NATIVE-AGENTS-FINAL"))
    );
    assert!(logs.iter().all(|span| span.trace_id == traces[0].trace_id));
    assert!(logs.iter().all(|span| {
        traces
            .iter()
            .any(|parent| parent.span_id == span.parent_span_id)
    }));
    let child = logs
        .iter()
        .find(|span| {
            span.normalized.output.contains("NATIVE-READER")
                && span
                    .attributes
                    .get("query_source")
                    .is_some_and(|source| source.starts_with("agent:"))
        })
        .unwrap();
    let execution = traces
        .iter()
        .find(|span| span.span_id == child.parent_span_id)
        .unwrap();
    assert_eq!(execution.name, "claude_code.tool.execution");
    assert!(
        traces
            .iter()
            .any(|span| span.span_id == execution.parent_span_id && span.name == "Agent")
    );
    assert!(
        logs.iter().all(|span| span
            .attributes
            .get("query_source")
            .is_none_or(|source| !matches!(
                source.as_str(),
                "prompt_suggestion" | "generate_session_title"
            )))
    );
}

#[rstest]
#[case::tool_result("tool_result", "", false)]
#[case::complete_body("api_request_body", r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","is_error":true,"content":[{"type":"text","text":"exit 3 output"},{"type":"image","source":{"data":"PRIVATE_IMAGE"}}]}]}],"system":"PRIVATE_SYSTEM"}"#, false)]
#[case::headless_body("api_request_body", r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","is_error":true,"content":"exit 3 output"}]},{"role":"system","content":"PRIVATE_SYSTEM"}]}"#, false)]
#[case::missing_messages("api_request_body", r#"{}"#, true)]
#[case::wrong_content(
    "api_request_body",
    r#"{"messages":[{"role":"user","content":{}}]}"#,
    true
)]
#[case::unexpected_last_message(
    "api_request_body",
    r#"{"messages":[{"role":"assistant","content":"unexpected"}]}"#,
    true
)]
#[case::truncated_body("api_request_body", "{truncated", true)]
fn native_tool_logs_supply_arguments_and_results_without_fake_calls(
    #[case] event: &str,
    #[case] body: &str,
    #[case] warning: bool,
) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use prost::Message;
    let mut request = log_request("repl_main_thread");
    request.resource_logs[0].scope_logs[0].log_records[0].attributes = [
        ("event.name", event),
        ("query_source", "repl_main_thread"),
        ("body", body),
        ("tool_use_id", "call-1"),
        ("success", "false"),
        ("error", "exit 3"),
        (
            "tool_input",
            r#"{"command":"exit 3","description":"Expected failure"}"#,
        ),
    ]
    .into_iter()
    .map(|(key, text)| KeyValue {
        key: key.into(),
        value: Some(AnyValue {
            value: Some(Value::StringValue(text.into())),
        }),
        ..Default::default()
    })
    .collect();
    let spans = litellm_traces::decode_otlp_logs(&request.encode_to_vec(), None).unwrap();
    let span = &spans[0];
    assert_eq!(span.normalized.observation_type, ObservationType::Framework);
    assert_eq!(span.normalized.input_tokens, 0);
    if event == "tool_result" {
        assert_eq!(span.normalized.tool_call_id.as_deref(), Some("call-1"));
        assert!(span.normalized.input.contains("Expected failure"));
        assert_eq!(span.status_code, "STATUS_CODE_ERROR");
    } else {
        let output: serde_json::Value = serde_json::from_str(&span.normalized.output).unwrap();
        assert_eq!(output.get("warning").is_some(), warning);
        assert!(span.consumed_attributes.contains(&"body"));
        if !warning {
            assert_eq!(output["tool_results"][0]["id"], "call-1");
            assert!(
                output["tool_results"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("exit 3 output")
            );
            assert!(!span.normalized.output.contains("PRIVATE"));
        }
    }
}

#[rstest]
fn interactive_claude_body_export_retains_failed_command_stdout() {
    let spans = litellm_traces::decode_otlp_logs(
        include_bytes!("fixtures/claude_code_native_tool_result.json"),
        Some("application/json"),
    )
    .unwrap();
    let output: serde_json::Value = serde_json::from_str(&spans[0].normalized.output).unwrap();
    let failed = output["tool_results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|result| result["is_error"] == true)
        .unwrap();
    assert_eq!(failed["content"], "Exit code 3\nRAW-EXPECTED");
}

#[rstest]
#[case::new_prompt(serde_json::json!({"role":"user", "content":"Continue"}))]
#[case::new_blocks(serde_json::json!({"role":"user", "content":[{"type":"text", "text":"Continue"}]}))]
fn native_body_exports_do_not_replay_old_tool_results(#[case] final_message: serde_json::Value) {
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    let mut request = log_request("repl_main_thread");
    let body = serde_json::json!({"messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old-call", "content": "OLD_RESULT"}]},
        {"role": "assistant", "content": "Done"}, final_message,
        {"role": "system", "content": "PRIVATE_SYSTEM"}
    ]}).to_string();
    request.resource_logs[0].scope_logs[0].log_records[0].attributes = [
        ("event.name", "api_request_body"),
        ("query_source", "repl_main_thread"),
        ("body", body.as_str()),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.into(),
        value: Some(AnyValue {
            value: Some(Value::StringValue(value.into())),
        }),
        ..Default::default()
    })
    .collect();
    let spans = litellm_traces::decode_otlp_logs(
        &serde_json::to_vec(&request).unwrap(),
        Some("application/json"),
    )
    .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&spans[0].normalized.output).unwrap(),
        serde_json::json!({"tool_results":[]})
    );
}
