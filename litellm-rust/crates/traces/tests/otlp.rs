use flate2::{Compression, write::GzEncoder};
use litellm_traces::decode_otlp;
use rstest::rstest;
use std::io::Write;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"
);

#[rstest]
#[case::json(FIXTURE, Some("application/json"), None)]
#[case::gzip_json(FIXTURE, Some("application/json"), Some("gzip"))]
fn decodes_neutral_spans(
    #[case] body: &[u8],
    #[case] content_type: Option<&str>,
    #[case] content_encoding: Option<&str>,
) {
    let payload = if content_encoding == Some("gzip") {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(body).expect("gzip input");
        encoder.finish().expect("gzip payload")
    } else {
        body.to_vec()
    };
    let spans = decode_otlp(&payload, content_type, content_encoding, 8 * 1024 * 1024)
        .expect("valid OTLP export");
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
fn rejects_invalid_or_oversized_payload(
    #[case] body: &[u8],
    #[case] content_type: Option<&str>,
    #[case] limit: usize,
) {
    assert!(decode_otlp(body, content_type, None, limit).is_err());
}
