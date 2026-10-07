use litellm_tracing::{
    Logger, Metadata, PayloadShape, Record, ShapeLimits, Sink,
    payload::{self, PayloadEvent, PayloadOutcome, PayloadStage},
};
use rstest::rstest;
use serde_json::{Value, json};
use std::sync::mpsc;

struct Capture(mpsc::Sender<Value>);
impl Sink for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn emit(&self, record: &Record) {
        if record.metadata.target() == payload::TARGET {
            self.0.send(Value::Object(record.fields.clone())).unwrap();
        }
    }
}

#[rstest]
fn projection_drops_inherited_payloads_and_context_values() {
    let (sender, records) = mpsc::channel();
    let logger = Logger::new(Capture(sender)).with_fields(
        json!({"private": "context-secret"})
            .as_object()
            .unwrap()
            .clone(),
    );
    logger.scope(|| {
        let span = tracing::info_span!(
            "parent",
            request_body = "private-prompt",
            model = "private-model"
        );
        let _entered = span.enter();
        PayloadEvent {
            stage: PayloadStage::RequestSent,
            shape: PayloadShape::extract(
                &json!({"messages": [{"content": "secret"}]}),
                ShapeLimits::default(),
            ),
            capture_id: "call-id".into(),
            outcome: Some(PayloadOutcome::Success),
        }
        .emit();
    });
    let record = records.try_recv().unwrap();
    assert_eq!(
        record,
        json!({"event.name":"llm.payload.shape", "schema_version":1, "trace_id":"call-id", "payload.stage":"provider.request.sent", "payload.field_paths":["$['messages']", "$['messages'][*]['content']"], "payload.shape_truncated":false, "payload.outcome":"success"})
    );
    assert!(records.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn capture_correlates_stages_across_yields_and_retains_existing_trace_identity() {
    let (sender, records) = mpsc::channel();
    let logger = Logger::new(Capture(sender)).with_fields(
        json!({"trace_id":"existing-id"})
            .as_object()
            .unwrap()
            .clone(),
    );
    logger
        .instrument(payload::capture(async {
            payload::record(PayloadStage::RequestReceived, &json!({"input":"private"}));
            tokio::task::yield_now().await;
            payload::record(
                PayloadStage::ResponseNormalized,
                &json!({"output":"private"}),
            );
        }))
        .await;
    let events: Vec<_> = records.try_iter().collect();
    assert_eq!(events.len(), 2);
    assert!(
        events
            .iter()
            .all(|event| event["trace_id"] == "existing-id")
    );
}

#[rstest]
#[case::empty_id("")]
#[case::long_id(
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)]
fn invalid_capture_ids_emit_nothing(#[case] id: &str) {
    let (sender, records) = mpsc::channel();
    Logger::new(Capture(sender)).scope(|| {
        PayloadEvent {
            stage: PayloadStage::RequestReceived,
            shape: PayloadShape::default(),
            capture_id: id.to_owned(),
            outcome: None,
        }
        .emit()
    });
    assert!(records.try_recv().is_err());
}
