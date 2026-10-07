mod support;

use litellm_tracing::{Logger, Metadata, Record, Sink, payload};
use rstest::rstest;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<Value>>>);
impl Sink for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn emit(&self, record: &Record) {
        if record.metadata.target() == payload::TARGET {
            self.0
                .lock()
                .unwrap()
                .push(Value::Object(record.fields.clone()));
        }
    }
}

#[rstest]
#[case::chat("/v1/chat/completions", "anthropic/test-model", json!({"messages":[{"role":"user","content":"private-input"}],"max_tokens":16}), json!({"id":"private-id","model":"test-model","content":[{"type":"text","text":"private-output"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}), "$['messages'][*]['content']", "$['content'][*]['text']", "$['choices'][*]['message']['content']")]
#[case::messages("/v1/messages", "anthropic/test-model", json!({"messages":[{"role":"user","content":"private-input"}],"max_tokens":16}), json!({"id":"private-id","type":"message","role":"assistant","model":"test-model","content":[{"type":"text","text":"private-output"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}), "$['messages'][*]['content']", "$['content'][*]['text']", "$['content'][*]['text']")]
#[case::responses("/v1/responses", "openai/test-model", json!({"input":"private-input"}), json!({"id":"private-id","model":"test-model","output":[{"type":"message","content":[{"type":"output_text","text":"private-output"}]}]}), "$['input']", "$['output'][*]['content'][*]['text']", "$['output'][*]['content'][*]['text']")]
#[tokio::test]
async fn gateway_and_core_share_one_capture_without_duplicate_input(
    #[case] route: &str,
    #[case] model: &str,
    #[case] input: Value,
    #[case] output: Value,
    #[case] request_path: &str,
    #[case] received_path: &str,
    #[case] normalized_path: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(output))
        .expect(1)
        .mount(&upstream)
        .await;
    let mut input = input.as_object().unwrap().clone();
    input.insert("model".into(), "public/model".into());
    let capture = Capture::default();
    let response = Logger::new(capture.clone())
        .instrument(support::post(
            support::app(model, &upstream.uri()),
            route,
            Value::Object(input),
        ))
        .await;
    assert_eq!(response.status(), 200, "{}", support::json(response).await);
    let events = capture.0.lock().unwrap();
    let stages: Vec<_> = events
        .iter()
        .map(|event| event["payload.stage"].as_str().unwrap())
        .collect();
    assert_eq!(
        stages,
        [
            "litellm.request.received",
            "provider.request.transformed",
            "provider.request.sent",
            "provider.response.received",
            "litellm.response.normalized"
        ]
    );
    assert!(
        events
            .iter()
            .all(|event| event["trace_id"] == events[0]["trace_id"])
    );
    assert!(
        events[0]["payload.field_paths"]
            .as_array()
            .unwrap()
            .contains(&request_path.into())
    );
    assert!(
        events[3]["payload.field_paths"]
            .as_array()
            .unwrap()
            .contains(&received_path.into())
    );
    assert!(
        events[4]["payload.field_paths"]
            .as_array()
            .unwrap()
            .contains(&normalized_path.into())
    );
    let encoded = serde_json::to_string(&*events).unwrap();
    assert!(!encoded.contains("private-"));
    assert!(!encoded.contains("public/model"));
}

#[rstest]
#[case::messages("/v1/messages", "anthropic/test-model", json!({"messages":[{"role":"user","content":"private-input"}],"max_tokens":16,"stream":true}), "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"private-output\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", "$['delta']['text']")]
#[case::responses("/v1/responses", "openai/test-model", json!({"input":"private-input","stream":true}), "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"private-output\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"tokens\":1}}}\n\n", "$['response']['usage']['tokens']")]
#[tokio::test]
async fn returned_http_streams_keep_capture_context_and_exact_response_bytes(
    #[case] route: &str,
    #[case] model: &str,
    #[case] input: Value,
    #[case] output: &str,
    #[case] normalized_path: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(output),
        )
        .expect(1)
        .mount(&upstream)
        .await;
    let mut input = input.as_object().unwrap().clone();
    input.insert("model".into(), "public/model".into());
    let capture = Capture::default();
    let response = Logger::new(capture.clone())
        .instrument(support::post(
            support::app(model, &upstream.uri()),
            route,
            Value::Object(input),
        ))
        .await;
    assert_eq!(response.status(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    assert_eq!(bytes.as_ref(), output.as_bytes());
    let events = capture.0.lock().unwrap();
    assert_eq!(events.len(), 5);
    assert!(
        events
            .iter()
            .all(|event| event["trace_id"] == events[0]["trace_id"])
    );
    assert_eq!(events[3]["payload.stage"], "provider.response.received");
    assert_eq!(events[4]["payload.stage"], "litellm.response.normalized");
    assert_eq!(events[3]["payload.outcome"], "success");
    assert_eq!(events[4]["payload.outcome"], "success");
    assert!(
        events[4]["payload.field_paths"]
            .as_array()
            .unwrap()
            .contains(&normalized_path.into())
    );
    assert!(
        !serde_json::to_string(&*events)
            .unwrap()
            .contains("private-")
    );
}

#[rstest]
#[case::chat("/v1/chat/completions", "anthropic/test-model", json!({"messages":[{"role":"user","content":"private-input"}],"max_tokens":16}))]
#[case::messages("/v1/messages", "anthropic/test-model", json!({"messages":[{"role":"user","content":"private-input"}],"max_tokens":16}))]
#[case::responses("/v1/responses", "openai/test-model", json!({"input":"private-input"}))]
#[tokio::test]
async fn malformed_provider_json_emits_a_truncated_received_shape_without_a_normalized_result(
    #[case] route: &str,
    #[case] model: &str,
    #[case] input: Value,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("private-invalid-json"))
        .expect(1)
        .mount(&upstream)
        .await;
    let mut input = input.as_object().unwrap().clone();
    input.insert("model".into(), "public/model".into());
    let capture = Capture::default();
    let response = Logger::new(capture.clone())
        .instrument(support::post(
            support::app(model, &upstream.uri()),
            route,
            Value::Object(input),
        ))
        .await;
    assert!(response.status().is_server_error());
    let events = capture.0.lock().unwrap();
    assert_eq!(events.len(), 4);
    assert_eq!(events[3]["payload.stage"], "provider.response.received");
    assert_eq!(events[3]["payload.shape_truncated"], true);
    assert_eq!(events[3]["payload.field_paths"], json!([]));
    assert!(
        !serde_json::to_string(&*events)
            .unwrap()
            .contains("private-")
    );
}
