use litellm_traces::decode_otlp;
use rstest::rstest;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"
);

#[rstest]
#[case::json(FIXTURE, Some("application/json"))]
fn decodes_neutral_spans(#[case] body: &[u8], #[case] content_type: Option<&str>) {
    let spans = decode_otlp(body, content_type, 8 * 1024 * 1024).expect("valid OTLP export");
    assert_eq!(spans.len(), 6);
    assert_eq!(spans[0].trace_id, "4bad42b84e9de3ba46fc870185f8f023");
    assert_eq!(spans[0].resource_attributes["service.name"], "agent-demo");
    assert_eq!(spans[0].scope_name, "langsmith");
    assert!(
        spans
            .iter()
            .any(|span| span.attributes.contains_key("gen_ai.prompt"))
    );
}

#[rstest]
#[case::invalid(b"not protobuf", None, 8 * 1024 * 1024)]
#[case::too_large(FIXTURE, Some("application/json"), 1)]
#[case::unsupported(FIXTURE, Some("text/plain"), 8 * 1024 * 1024)]
fn rejects_invalid_or_oversized_payload(
    #[case] body: &[u8],
    #[case] content_type: Option<&str>,
    #[case] limit: usize,
) {
    assert!(decode_otlp(body, content_type, limit).is_err());
}

#[rstest]
fn shared_resource_attributes_count_toward_decode_budget() {
    use opentelemetry_proto::tonic::{
        collector::trace::v1::ExportTraceServiceRequest,
        common::v1::{AnyValue, KeyValue, any_value::Value},
        resource::v1::Resource,
        trace::v1::{ResourceSpans, ScopeSpans, Span},
    };
    use prost::Message;

    let request = ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            resource: Some(Resource {
                attributes: vec![KeyValue {
                    key: "shared".into(),
                    value: Some(AnyValue {
                        value: Some(Value::StringValue("x".repeat(4096))),
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            scope_spans: vec![ScopeSpans {
                spans: vec![Span::default(); 100],
                ..Default::default()
            }],
            ..Default::default()
        }],
    };
    let body = request.encode_to_vec();
    assert!(body.len() < 16 * 1024);
    assert!(matches!(
        decode_otlp(&body, None, 16 * 1024),
        Err(litellm_traces::DecodeError::TooLarge)
    ));
}
