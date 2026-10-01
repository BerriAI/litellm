use litellm_traces::{OTLP_DEFAULT_MAX_BODY_BYTES, decode_otlp};
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
    assert_eq!(spans[0].scope_name, "langsmith");
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
#[case::invalid(b"not protobuf".to_vec(), None)]
#[case::too_large(vec![b' '; OTLP_DEFAULT_MAX_BODY_BYTES + 1], Some("application/json"))]
fn rejects_invalid_or_oversized_payload(#[case] body: Vec<u8>, #[case] content_type: Option<&str>) {
    assert!(decode_otlp(&body, content_type).is_err());
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
fn resource_fanout_is_charged_before_copying(span: opentelemetry_proto::tonic::trace::v1::Span) {
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
    let body = request.encode_to_vec();
    assert!(body.len() < OTLP_DEFAULT_MAX_BODY_BYTES);
    assert!(matches!(
        decode_otlp(&body, None),
        Err(litellm_traces::DecodeError::TooLarge)
    ));
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
    assert!(body.len() < OTLP_DEFAULT_MAX_BODY_BYTES);
    assert!(matches!(
        decode_otlp(&body, None),
        Err(litellm_traces::DecodeError::TooLarge)
    ));
}
