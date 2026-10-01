use flate2::{Compression, write::GzEncoder};
use litellm_traces::{ObservationType, decode_otlp};
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

#[rstest]
fn normalizes_langsmith_fixture() {
    let spans = decode_otlp(FIXTURE, Some("application/json"), None, 8 * 1024 * 1024)
        .expect("valid OTLP export");
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
