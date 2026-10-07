use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_inference::payload::observe_sse;
use litellm_tracing::{
    Logger, Metadata, Record, Sink,
    payload::{self, PayloadStage},
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
#[case::whole(1000)]
#[case::fragmented(3)]
#[case::every_byte(1)]
#[tokio::test]
async fn sse_shapes_ignore_transport_fragmentation_and_preserve_exact_chunks(#[case] width: usize) {
    let (sender, records) = mpsc::channel();
    let logger = Logger::new(Capture(sender));
    let body = b"data: {\"delta\":{\"text\":\"private\"}}\r\n\r\ndata: {\"usage\":{\"tokens\":2}}\n\ndata: [DONE]\n\n";
    let original: Vec<_> = body.chunks(width).map(Bytes::copy_from_slice).collect();
    let expected = original.clone();
    let observed = logger
        .instrument(payload::capture(async {
            let chunks = stream::iter(original.into_iter().map(Ok::<_, ()>)).boxed();
            observe_sse(chunks, PayloadStage::ResponseReceived)
                .collect::<Vec<_>>()
                .await
        }))
        .await;
    assert_eq!(observed, expected.into_iter().map(Ok).collect::<Vec<_>>());
    let event = records.try_recv().unwrap();
    assert_eq!(
        event["payload.field_paths"],
        json!([
            "$['delta']",
            "$['delta']['text']",
            "$['usage']",
            "$['usage']['tokens']"
        ])
    );
    assert_eq!(event["payload.outcome"], "success");
    assert_eq!(event["payload.shape_truncated"], false);
    assert!(!event.to_string().contains("private"));
    assert!(records.try_recv().is_err());
}

#[rstest]
#[case::failure(true, "failure")]
#[case::cancellation(false, "cancelled")]
#[tokio::test]
async fn partial_streams_emit_one_terminal_summary_even_when_dropped_elsewhere(
    #[case] fail: bool,
    #[case] outcome: &str,
) {
    let (sender, records) = mpsc::channel();
    let logger = Logger::new(Capture(sender));
    let mut chunks = logger
        .instrument(payload::capture(async {
            let source = stream::iter([
                Ok(Bytes::from_static(
                    b"data: {\"delta\":{\"text\":\"private\"}}\n\n",
                )),
                Err(()),
            ])
            .boxed();
            observe_sse(source, PayloadStage::ResponseReceived)
        }))
        .await;
    assert!(chunks.next().await.unwrap().is_ok());
    if fail {
        assert!(chunks.next().await.unwrap().is_err());
    }
    Logger::default().scope(|| drop(chunks));
    let event = records.try_recv().unwrap();
    assert_eq!(event["payload.outcome"], outcome);
    assert_eq!(
        event["payload.field_paths"],
        json!(["$['delta']", "$['delta']['text']"])
    );
    assert!(records.try_recv().is_err());
}

#[rstest]
#[case::unfinished(b"data: {\"private\":".as_slice())]
#[case::invalid(b"data: invalid\n\n".as_slice())]
#[tokio::test]
async fn incomplete_or_invalid_json_discards_partial_paths(#[case] bytes: &'static [u8]) {
    let (sender, records) = mpsc::channel();
    Logger::new(Capture(sender))
        .instrument(payload::capture(async {
            let chunks = stream::iter([Ok::<_, ()>(Bytes::from_static(bytes))]).boxed();
            let _ = observe_sse(chunks, PayloadStage::ResponseReceived)
                .collect::<Vec<_>>()
                .await;
        }))
        .await;
    let event = records.try_recv().unwrap();
    assert_eq!(event["payload.field_paths"], json!([]));
    assert_eq!(event["payload.shape_truncated"], true);
}

#[rstest]
#[case::bytes(Bytes::from(vec![b'a'; 1_048_577]))]
#[case::events(Bytes::from("data: {}\n\n".repeat(4097)))]
#[tokio::test]
async fn stream_processing_limits_preserve_delivery_and_discard_diagnostic_paths(
    #[case] chunk: Bytes,
) {
    let (sender, records) = mpsc::channel();
    let expected = chunk.clone();
    let delivered = Logger::new(Capture(sender))
        .instrument(payload::capture(async {
            observe_sse(
                stream::iter([Ok::<_, ()>(chunk)]).boxed(),
                PayloadStage::ResponseReceived,
            )
            .collect::<Vec<_>>()
            .await
        }))
        .await;
    assert_eq!(delivered, [Ok(expected)]);
    let event = records.try_recv().unwrap();
    assert_eq!(event["payload.shape_truncated"], true);
    assert_eq!(event["payload.field_paths"], json!([]));
    assert_eq!(event["payload.outcome"], "success");
}
