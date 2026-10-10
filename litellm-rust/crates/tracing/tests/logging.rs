use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_tracing::{ByteChunk, Level, Logger, Metadata, Record, Sink, info, warn};
use rstest::rstest;
use serde_json::{Value, json};

struct Output {
    enabled: Arc<AtomicBool>,
    sender: mpsc::Sender<(String, Value, Level, &'static str, Option<u32>)>,
}

impl Sink for Output {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn emit(&self, record: &Record) {
        self.sender
            .send((
                record.message.clone(),
                Value::Object(record.fields.clone()),
                *record.metadata.level(),
                record.metadata.target(),
                record.metadata.line(),
            ))
            .unwrap();
        Logger::default().scope(|| warn!("a sink must not recursively emit"));
    }
}

fn emit() {
    warn!(
        attempt = 3_u64,
        elapsed = 1.5,
        retry = true,
        reason = "timeout",
        "retry {}",
        3
    );
}

#[test]
fn records_preserve_fields_metadata_and_dynamic_filtering_without_recursion() {
    let (sender, receiver) = mpsc::channel();
    let enabled = Arc::new(AtomicBool::new(false));
    let logger = Logger::new(Output {
        enabled: enabled.clone(),
        sender,
    });
    logger.scope(emit);
    assert!(receiver.try_recv().is_err());
    enabled.store(true, Ordering::Relaxed);
    logger.scope(emit);
    let (message, fields, level, target, line) = receiver.try_recv().unwrap();
    assert_eq!(message, "retry 3");
    assert_eq!(
        fields,
        json!({"attempt": 3, "elapsed": 1.5, "retry": true, "reason": "timeout"})
    );
    assert_eq!(level, Level::WARN);
    assert_eq!(target, module_path!());
    assert!(line.is_some());
    enabled.store(false, Ordering::Relaxed);
    logger.scope(emit);
    assert!(receiver.try_recv().is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_futures_keep_their_sinks_across_suspension_and_spawn() {
    let tasks = (0..2)
        .map(|id| {
            let (sender, receiver) = mpsc::channel();
            let logger = Logger::new(Output {
                enabled: Arc::new(AtomicBool::new(true)),
                sender,
            });
            let task = tokio::spawn(logger.instrument(async move {
                tokio::task::yield_now().await;
                info!(id, "worker");
            }));
            (id, task, receiver)
        })
        .collect::<Vec<_>>();
    for (id, task, receiver) in tasks {
        task.await.unwrap();
        let (message, fields, level, _, _) = receiver.try_recv().unwrap();
        assert_eq!(message, "worker");
        assert_eq!(fields, json!({"id": id}));
        assert_eq!(level, Level::INFO);
        assert!(receiver.try_recv().is_err());
    }
}

#[test]
fn nested_scopes_restore_the_previous_sink() {
    let (outer_sender, outer) = mpsc::channel();
    let (inner_sender, inner) = mpsc::channel();
    let logger = |sender| {
        Logger::new(Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender,
        })
    };
    let outside = logger(outer_sender);
    let inside = logger(inner_sender);
    outside.scope(|| {
        info!("before");
        inside.scope(|| info!("inside"));
        info!("after");
    });
    assert_eq!(
        outer.try_iter().map(|event| event.0).collect::<Vec<_>>(),
        ["before", "after"]
    );
    assert_eq!(
        inner.try_iter().map(|event| event.0).collect::<Vec<_>>(),
        ["inside"]
    );
}

#[rstest]
#[case::utf8(b"event: message_stop\n\n", "utf8")]
#[case::binary(&[0xff, 0x00, 0x80], "base64")]
fn byte_chunk_logging_preserves_exact_bytes(#[case] bytes: &[u8], #[case] encoding: &str) {
    let chunk = ByteChunk::new(bytes);
    assert_eq!(chunk.encoding(), encoding);
    let text = chunk.to_string();
    let recovered = match encoding {
        "utf8" => text.into_bytes(),
        "base64" => STANDARD.decode(text).unwrap(),
        _ => unreachable!(),
    };
    assert_eq!(recovered, bytes);
}

#[rstest]
fn spans_inherit_fields_accept_updates_and_close_once() {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Output {
        enabled: Arc::new(AtomicBool::new(true)),
        sender,
    });
    logger.scope(|| {
        let route = tracing::info_span!(
            "route",
            route = "messages",
            model = "requested",
            message = "span message",
            outcome = tracing::field::Empty
        );
        route.in_scope(|| {
            let child = tracing::debug_span!(
                "provider",
                model = "resolved",
                status = tracing::field::Empty
            );
            child.record("status", 201_u64);
            child.in_scope(|| info!(model = "event", "response"));
            route.record("outcome", "success");
        });
    });
    let records = receiver.try_iter().collect::<Vec<_>>();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].0, "response");
    assert_eq!(
        records[0].1,
        json!({"route": "messages", "model": "event", "status": 201, "message": "span message"})
    );
    assert_eq!(records[1].1["span_name"], "provider");
    assert_eq!(records[1].1["model"], "resolved");
    assert_eq!(records[1].1["status"], 201);
    assert_eq!(records[2].1["span_name"], "route");
    assert_eq!(records[2].1["model"], "requested");
    assert_eq!(records[2].1["outcome"], "success");
    assert!(
        records[2].1["duration_ms"].as_f64().unwrap()
            >= records[1].1["duration_ms"].as_f64().unwrap()
    );
}

#[rstest]
fn sink_layers_filter_independently_and_recheck_span_delivery() {
    use tracing_subscriber::prelude::*;

    let (first_sender, first) = mpsc::channel();
    let (second_sender, second) = mpsc::channel();
    let enabled = Arc::new(AtomicBool::new(false));
    let subscriber = tracing_subscriber::Registry::default()
        .with(litellm_tracing::sink_layer(Output {
            enabled: enabled.clone(),
            sender: first_sender,
        }))
        .with(litellm_tracing::sink_layer(Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender: second_sender,
        }));
    tracing::subscriber::with_default(subscriber, || {
        info!("second only");
        enabled.store(true, Ordering::Relaxed);
        let span = tracing::info_span!("shared", route = "responses");
        span.in_scope(|| info!("both"));
        enabled.store(false, Ordering::Relaxed);
        drop(span);
    });
    assert_eq!(
        first.try_iter().map(|record| record.0).collect::<Vec<_>>(),
        ["both"]
    );
    let records = second.try_iter().collect::<Vec<_>>();
    assert_eq!(
        records
            .iter()
            .map(|record| record.0.as_str())
            .collect::<Vec<_>>(),
        ["second only", "both", "span closed"]
    );
    assert_eq!(records[1].1["route"], "responses");
    assert_eq!(records[2].1["route"], "responses");
}

#[rstest]
fn disabled_spans_do_not_evaluate_fields() {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Output {
        enabled: Arc::new(AtomicBool::new(false)),
        sender,
    });
    let evaluated = AtomicBool::new(false);
    logger.scope(|| {
        let _span = tracing::info_span!(
            "disabled",
            field = {
                evaluated.store(true, Ordering::Relaxed);
                "value"
            }
        );
        assert!(_span.is_disabled());
    });
    assert!(!evaluated.load(Ordering::Relaxed));
    assert!(receiver.try_recv().is_err());
}

#[tracing::instrument(name = "worker", skip_all, fields(id))]
async fn traced_worker(id: u64) {
    tracing::Span::current().record("id", id);
    tokio::task::yield_now().await;
    info!("inside worker");
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn instrumented_futures_keep_span_fields_in_their_captured_sink() {
    let tasks = (0..2)
        .map(|id| {
            let (sender, receiver) = mpsc::channel();
            let logger = Logger::new(Output {
                enabled: Arc::new(AtomicBool::new(true)),
                sender,
            });
            (
                id,
                tokio::spawn(logger.instrument(traced_worker(id))),
                receiver,
            )
        })
        .collect::<Vec<_>>();
    for (id, task, receiver) in tasks {
        task.await.unwrap();
        let records = receiver.try_iter().collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, "inside worker");
        assert_eq!(records[0].1["id"], id);
        assert_eq!(records[1].1["id"], id);
        assert_eq!(records[1].1["span_name"], "worker");
    }
}
