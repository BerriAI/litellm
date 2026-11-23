use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_tracing::{ByteChunk, Level, Logger, Metadata, Record, Scrubbed, Sink};
use rstest::rstest;
use serde_json::{Value, json};
use tracing::{info, warn};

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
fn a_span_that_closes_after_its_sink_stopped_listening_is_not_reported() {
    let (sender, receiver) = mpsc::channel();
    let enabled = Arc::new(AtomicBool::new(true));
    let logger = Logger::new(Output {
        enabled: enabled.clone(),
        sender,
    });
    logger.scope(|| {
        let span = tracing::info_span!("shared", route = "responses");
        span.in_scope(|| info!("inside"));
        enabled.store(false, Ordering::Relaxed);
        drop(span);
    });
    assert_eq!(
        receiver
            .try_iter()
            .map(|record| record.0)
            .collect::<Vec<_>>(),
        ["inside"]
    );
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

#[rstest]
#[case::bearer_in_the_message("Authorization: Bearer abcdefghijklmnop")]
#[case::url_userinfo_in_the_message("postgres://user:credential-value@database.internal/name")]
fn a_scrubbed_sink_never_sees_a_secret(#[case] secret: &str) {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Scrubbed::new(
        Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender,
        },
        16,
    ));

    logger.scope(|| {
        warn!(
            api_key = "credential-value",
            detail = secret,
            attempt = 3_u64,
            "upstream said {secret}"
        )
    });

    let (message, fields, ..) = receiver.try_recv().unwrap();
    let line = format!("{message} {fields}");
    assert!(!line.contains("credential-value"), "{line}");
    assert!(!line.contains("abcdefghijklmnop"), "{line}");
    assert!(message.starts_with("upstream said "), "{message}");
    assert_eq!(fields["api_key"], "REDACTED");
    assert_eq!(fields["attempt"], 3);
}

#[rstest]
fn a_scrubbed_sink_scrubs_a_logged_json_body_value_by_value() {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Scrubbed::new(
        Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender,
        },
        16,
    ));
    let pasted = json!({"password": "credential-value"}).to_string();
    let body = json!({
        "model": "claude-haiku",
        "max_tokens": 20,
        "api_key": "another-credential",
        "mcp_servers": [{"name": "docs", "authorization_token": "opaque-mcp-credential"}],
        "messages": [{"role": "user", "content": format!("my config is {pasted}")}],
    })
    .to_string();
    assert!(body.contains(r#"\"password\""#));

    logger.scope(|| warn!(chunk = body, "body chunk"));

    let (_, fields, ..) = receiver.try_recv().unwrap();
    let chunk: Value = serde_json::from_str(fields["chunk"].as_str().unwrap()).unwrap();
    assert!(!chunk.to_string().contains("credential"), "{chunk}");
    assert_eq!(chunk["model"], "claude-haiku");
    assert_eq!(chunk["max_tokens"], 20);
    assert_eq!(chunk["api_key"], "REDACTED");
    assert_eq!(chunk["mcp_servers"][0]["name"], "docs");
    assert_eq!(chunk["mcp_servers"][0]["authorization_token"], "REDACTED");
    assert_eq!(chunk["messages"][0]["role"], "user");
    assert!(
        chunk["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with("my config is ")
    );
}

#[rstest]
fn a_scrubbed_sink_scrubs_the_json_of_each_stream_event_and_a_secret_number() {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Scrubbed::new(
        Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender,
        },
        16,
    ));
    let pasted = json!({"password": "credential-value"}).to_string();
    let delta = json!({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": pasted},
    });
    let frame = format!(
        "event: content_block_delta\r\ndata: {delta}\r\n\r\nevent: ping\ndata: not json\n\n"
    );
    let body = json!({
        "password": 12345678,
        "max_tokens": 20,
        "input": {"sk-abcdefghijklmnopqrstuvwxyz": 1, "city": "Paris"},
    })
    .to_string();

    logger.scope(|| warn!(chunk = frame, body, max_tokens = 20_u64, "body chunk"));

    let (_, fields, ..) = receiver.try_recv().unwrap();
    let chunk = fields["chunk"].as_str().unwrap();
    assert!(!chunk.contains("credential-value"), "{chunk}");
    let lines: Vec<&str> = chunk.split("\n").collect();
    assert_eq!(lines[0], "event: content_block_delta\r");
    let data: Value =
        serde_json::from_str(lines[1].strip_prefix("data: ").unwrap().trim_end()).unwrap();
    assert_eq!(data["type"], "content_block_delta");
    assert_eq!(data["delta"]["type"], "text_delta");
    assert_eq!(&lines[2..], ["\r", "event: ping", "data: not json", "", ""]);
    let body: Value = serde_json::from_str(fields["body"].as_str().unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "password": "REDACTED",
            "max_tokens": 20,
            "input": {"REDACTED": "REDACTED", "city": "Paris"},
        })
    );
    assert_eq!(fields["max_tokens"], 20);
}

#[rstest]
fn a_scrubbed_sink_takes_a_key_length_set_after_it_is_installed() {
    let (sender, receiver) = mpsc::channel();
    let scrubbed = Scrubbed::new(
        Output {
            enabled: Arc::new(AtomicBool::new(true)),
            sender,
        },
        16,
    );
    let key_length = scrubbed.key_length();
    let logger = Logger::new(scrubbed);

    logger.scope(|| warn!("key sk-abcde rejected"));
    key_length.set(8);
    logger.scope(|| warn!("key sk-abcde rejected"));

    let messages: Vec<String> = receiver.try_iter().map(|record| record.0).collect();
    assert_eq!(messages, ["key sk-abcde rejected", "key REDACTED rejected"]);
}

#[rstest]
#[tokio::test]
async fn work_scoped_to_a_logger_within_a_span_inherits_the_spans_fields() {
    let (sender, receiver) = mpsc::channel();
    let logger = Logger::new(Output {
        enabled: Arc::new(AtomicBool::new(true)),
        sender,
    });
    let call = logger.within(|| tracing::info_span!("call", request = "first"));

    logger.scope(|| warn!("outside the call"));
    call.scope(|| warn!("sync inside the call"));
    call.instrument(async {
        tokio::task::yield_now().await;
        warn!(attempt = 2_u64, "async inside the call");
    })
    .await;

    let records: Vec<(String, Value)> = receiver
        .try_iter()
        .map(|(message, fields, ..)| (message, fields))
        .collect();
    assert_eq!(
        records,
        [
            ("outside the call".to_owned(), json!({})),
            (
                "sync inside the call".to_owned(),
                json!({"request": "first"})
            ),
            (
                "async inside the call".to_owned(),
                json!({"request": "first", "attempt": 2})
            ),
        ]
    );
}
