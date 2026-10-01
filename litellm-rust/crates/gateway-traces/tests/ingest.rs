use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use flate2::{Compression, write::GzEncoder};
use litellm_gateway_traces::{MAX_BODY_BYTES, SpanSink, router};
use litellm_traces::DecodedSpan;
use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest,
    trace::v1::{ResourceSpans, ScopeSpans, Span},
};
use prost::Message;
use rstest::rstest;
use tower::ServiceExt;

const FIXTURE: &[u8] = include_bytes!(
    "../../../../tests/test_litellm/tracing/fixtures/langsmith_deep_agent_export.json"
);

#[derive(Clone, Default)]
struct RecordingSink(Arc<Mutex<Vec<Vec<DecodedSpan>>>>);

struct FailingSink;

impl SpanSink for RecordingSink {
    async fn write(&self, spans: Vec<DecodedSpan>) -> Result<(), ()> {
        self.0.lock().unwrap().push(spans);
        Ok(())
    }
}

impl SpanSink for FailingSink {
    async fn write(&self, _: Vec<DecodedSpan>) -> Result<(), ()> {
        Err(())
    }
}

fn gzip(body: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(body).unwrap();
    encoder.finish().unwrap()
}

fn protobuf() -> Vec<u8> {
    ExportTraceServiceRequest {
        resource_spans: vec![ResourceSpans {
            scope_spans: vec![ScopeSpans {
                spans: vec![Span {
                    trace_id: vec![1; 16],
                    span_id: vec![2; 8],
                    start_time_unix_nano: 1,
                    end_time_unix_nano: 2,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
    .encode_to_vec()
}

#[rstest]
#[case::json(FIXTURE.to_vec(), None, StatusCode::OK)]
#[case::gzip(gzip(FIXTURE), Some("gzip"), StatusCode::OK)]
#[case::missing_header(gzip(FIXTURE), None, StatusCode::BAD_REQUEST)]
#[case::unsupported(FIXTURE.to_vec(), Some("br"), StatusCode::BAD_REQUEST)]
#[tokio::test]
async fn post_decodes_only_declared_content_encoding(
    #[case] body: Vec<u8>,
    #[case] encoding: Option<&str>,
    #[case] status: StatusCode,
) {
    let sink = RecordingSink::default();
    let mut request = Request::post("/v1/traces").header("content-type", "application/json");
    if let Some(encoding) = encoding {
        request = request.header("content-encoding", encoding);
    }
    let response = router(sink.clone())
        .oneshot(request.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), status);
    let calls = sink.0.lock().unwrap();
    assert_eq!(calls.len(), usize::from(status == StatusCode::OK));
    if status == StatusCode::OK {
        assert_eq!(calls[0][0].trace_id, "4bad42b84e9de3ba46fc870185f8f023");
    }
}

#[rstest]
#[tokio::test]
async fn post_decodes_protobuf() {
    let sink = RecordingSink::default();
    let response = router(sink.clone())
        .oneshot(
            Request::post("/v1/traces")
                .header("content-type", "application/x-protobuf")
                .body(Body::from(protobuf()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(sink.0.lock().unwrap()[0][0].trace_id, "01".repeat(16));
}

#[rstest]
#[tokio::test]
async fn post_does_not_acknowledge_a_failed_write() {
    let response = router(FailingSink)
        .oneshot(
            Request::post("/v1/traces")
                .header("content-type", "application/json")
                .body(Body::from(FIXTURE))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[rstest]
#[tokio::test]
async fn post_rejects_body_over_the_wire_limit() {
    let sink = RecordingSink::default();
    let response = router(sink.clone())
        .oneshot(
            Request::post("/v1/traces")
                .header("content-type", "application/json")
                .body(Body::from(vec![b' '; MAX_BODY_BYTES + 1]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(sink.0.lock().unwrap().is_empty());
}
