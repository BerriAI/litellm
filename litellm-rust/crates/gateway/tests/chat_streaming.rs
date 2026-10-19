//! `stream: true` chat completions for deployments chat translates (Anthropic, Bedrock
//! Converse): the OpenAI chunks the client gets, closed by `[DONE]`, and the usage and cost
//! the request is recorded with, whether or not the client asked to see the usage.

use std::sync::{Arc, Mutex};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode, header::CONTENT_TYPE},
    response::IntoResponse,
};
use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_framer::aws_event_stream::{AwsEventStreamCodec, Header, HeaderValue, Message};
use litellm_gateway_hooks::{Facts, Observer, Record};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::codec::Encoder;
use tower::ServiceExt;

/// Input, output, cache read and cache write prices of every deployment.
const RATES: (f64, f64, f64, f64) = (1.0, 2.0, 0.1, 1.25);

fn anthropic_events(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect()
}

fn anthropic_stream() -> String {
    anthropic_events(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x-20250101", "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 10, "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 12}}),
        json!({"type": "message_stop"}),
    ])
}

fn bedrock_frame(kind: &str, name: &str, payload: Value, wire: &mut bytes::BytesMut) {
    let type_header = match kind {
        "exception" => ":exception-type",
        _ => ":event-type",
    };
    let message = Message::new(Bytes::from(payload.to_string()))
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String(kind.to_owned().into()),
        ))
        .add_header(Header::new(
            type_header,
            HeaderValue::String(name.to_owned().into()),
        ));
    AwsEventStreamCodec.encode(message, wire).unwrap();
}

fn bedrock_stream() -> Bytes {
    let mut wire = bytes::BytesMut::new();
    for (name, payload) in [
        ("messageStart", json!({"role": "assistant"})),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "Hello"}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": " world"}}),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "end_turn"})),
        (
            "metadata",
            json!({"usage": {"inputTokens": 10, "outputTokens": 12, "totalTokens": 48, "cacheReadInputTokens": 20, "cacheWriteInputTokens": 6}, "metrics": {"latencyMs": 9}}),
        ),
    ] {
        bedrock_frame("event", name, payload, &mut wire);
    }
    wire.freeze()
}

fn bedrock_cut_stream() -> Bytes {
    let mut wire = bytes::BytesMut::new();
    bedrock_frame(
        "event",
        "messageStart",
        json!({"role": "assistant"}),
        &mut wire,
    );
    bedrock_frame(
        "event",
        "contentBlockDelta",
        json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
        &mut wire,
    );
    bedrock_frame(
        "exception",
        "throttlingException",
        json!({"message": "Too many requests"}),
        &mut wire,
    );
    wire.freeze()
}

#[derive(Clone, Debug)]
struct Sent {
    path: String,
    body: Value,
}

type Calls = Arc<Mutex<Vec<Sent>>>;

/// Answers by credential: each deployment's key picks what its provider does.
async fn upstream(calls: Calls) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let calls = calls.clone();
        async move {
            let (parts, body) = request.into_parts();
            let credential = parts
                .headers
                .get("x-api-key")
                .or_else(|| parts.headers.get("authorization"))
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let body: Value =
                serde_json::from_slice(&to_bytes(body, 1 << 20).await.unwrap()).unwrap();
            calls.lock().unwrap().push(Sent {
                path: parts.uri.path().to_owned(),
                body,
            });
            let sse = [(CONTENT_TYPE, "text/event-stream")];
            let event_stream = [(CONTENT_TYPE, "application/vnd.amazon.eventstream")];
            match credential.as_str() {
                "sk-anthropic" => (sse, anthropic_stream()).into_response(),
                "sk-anthropic-error-first" => (
                    sse,
                    anthropic_events(&[
                        json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "claude-x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 1}}}),
                        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
                    ]),
                )
                    .into_response(),
                "sk-anthropic-overloaded" => (
                    StatusCode::from_u16(529).unwrap(),
                    [(CONTENT_TYPE, "application/json")],
                    r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
                )
                    .into_response(),
                "Bearer bedrock-token" => (event_stream, bedrock_stream()).into_response(),
                "Bearer bedrock-cut" => (event_stream, bedrock_cut_stream()).into_response(),
                other => (StatusCode::UNAUTHORIZED, format!("unexpected credential {other}"))
                    .into_response(),
            }
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

#[derive(Debug)]
struct Seen {
    deployment: Option<String>,
    status: u16,
    usage: Option<litellm_gateway_hooks::Usage>,
    cost: Option<f64>,
    first_byte: bool,
}

struct Collect(Mutex<Vec<Seen>>);

impl Observer for Collect {
    fn name(&self) -> &'static str {
        "collect"
    }

    fn reads(&self) -> Facts {
        Facts::DEPLOYMENT | Facts::USAGE | Facts::COST
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.0.lock().unwrap().push(Seen {
                deployment: record
                    .outcome
                    .deployment
                    .as_ref()
                    .map(|deployment| deployment.id.to_string()),
                status: record.outcome.status,
                usage: record.outcome.usage,
                cost: record.outcome.cost,
                first_byte: record.outcome.time_to_first_byte.is_some(),
            });
        })
    }
}

struct Harness {
    app: Router,
    inference: Arc<litellm_gateway_inference::Gateway>,
    calls: Calls,
    seen: Arc<Collect>,
}

fn deployment(name: &str, id: &str, model: &str, upstream: &str, key: &str) -> String {
    let (input, output, cache_read, cache_write) = RATES;
    format!(
        "  - model_name: {name}
    litellm_params:
      model: {model}
      api_base: {upstream}
      api_key: {key}
      input_cost_per_token: {input}
      output_cost_per_token: {output}
      cache_read_input_token_cost: {cache_read}
      cache_creation_input_token_cost: {cache_write}
    model_info:
      id: {id}
"
    )
}

async fn harness() -> Harness {
    let calls = Calls::default();
    let upstream = upstream(calls.clone()).await;
    let bedrock = "bedrock/anthropic.claude-x-v1:0";
    let models = [
        deployment(
            "claude",
            "claude-1",
            "anthropic/claude-x",
            &upstream,
            "sk-anthropic",
        ),
        deployment("bedrock", "bedrock-1", bedrock, &upstream, "bedrock-token"),
        deployment(
            "bedrock-cut",
            "bedrock-2",
            bedrock,
            &upstream,
            "bedrock-cut",
        ),
        deployment(
            "overloaded",
            "claude-2",
            "anthropic/claude-x",
            &upstream,
            "sk-anthropic-overloaded",
        ),
        deployment(
            "fallback",
            "claude-3",
            "anthropic/claude-x",
            &upstream,
            "sk-anthropic-error-first",
        ),
        deployment("fallback", "bedrock-3", bedrock, &upstream, "bedrock-token"),
    ]
    .concat();
    let config = Config::from_yaml(&format!(
        "model_list:
{models}router_settings:
  num_retries: 1
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let seen = Arc::new(Collect(Mutex::new(Vec::new())));
    litellm_gateway::hooks::install(&inference, &config, Vec::new(), vec![seen.clone()])
        .await
        .unwrap();
    let app = litellm_gateway::router(inference.clone(), &config, None, None);
    Harness {
        app,
        inference,
        calls,
        seen,
    }
}

async fn send(app: &Router, body: Value) -> (u16, Option<String>, String) {
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-master")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (
        status,
        content_type,
        String::from_utf8(body.to_vec()).unwrap(),
    )
}

fn request(model: &str, include_usage: bool) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    if include_usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

/// The JSON of each `data:` frame, and whether the stream closed with `[DONE]`.
fn frames(body: &str) -> (Vec<Value>, bool) {
    let mut done = false;
    let frames = body
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .filter_map(|frame| {
            let data = frame.strip_prefix("data: ").expect("bare data frames");
            match data {
                "[DONE]" => {
                    done = true;
                    None
                }
                data => Some(serde_json::from_str(data).unwrap()),
            }
        })
        .collect();
    (frames, done)
}

fn usage_36_12() -> Value {
    json!({"completion_tokens": 12, "prompt_tokens": 36, "total_tokens": 48,
        "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 12},
        "prompt_tokens_details": {"cached_tokens": 20, "text_tokens": 10, "cache_write_tokens": 6, "cache_creation_tokens": 6},
        "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20})
}

fn expected_cost() -> f64 {
    let (input, output, cache_read, cache_write) = RATES;
    10.0 * input + 12.0 * output + 20.0 * cache_read + 6.0 * cache_write
}

async fn seen(harness: &Harness) -> Vec<Seen> {
    litellm_gateway::hooks::close(&harness.inference).await;
    std::mem::take(&mut *harness.seen.0.lock().unwrap())
}

#[rstest]
#[case::anthropic("claude", "claude-1", "/v1/messages")]
#[case::bedrock(
    "bedrock",
    "bedrock-1",
    "/model/anthropic.claude-x-v1:0/converse-stream"
)]
#[tokio::test]
async fn a_translated_stream_sends_openai_chunks_and_records_its_usage(
    #[case] model: &str,
    #[case] id: &str,
    #[case] path: &str,
    #[values(true, false)] include_usage: bool,
) {
    let harness = harness().await;
    let (status, content_type, body) = send(&harness.app, request(model, include_usage)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(content_type.as_deref(), Some("text/event-stream"));

    let (frames, done) = frames(&body);
    assert!(done, "{body}");
    let id_of = |frame: &Value| frame["id"].clone();
    assert!(frames.iter().all(|frame| id_of(frame) == id_of(&frames[0])));
    assert!(
        frames
            .iter()
            .all(|frame| frame["object"] == "chat.completion.chunk")
    );
    assert!(frames.iter().all(|frame| frame["model"] == model), "{body}");
    let choices: Vec<Value> = frames
        .iter()
        .map(|frame| frame["choices"].clone())
        .collect();
    let mut expected = vec![
        json!([{"index": 0, "delta": {"content": "Hello", "role": "assistant"}}]),
        json!([{"index": 0, "delta": {"content": " world"}}]),
        json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
    ];
    if include_usage {
        expected.push(json!([{"index": 0, "delta": {}}]));
    }
    assert_eq!(choices, expected, "{body}");
    let usages: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame.get("usage"))
        .collect();
    match include_usage {
        true => assert_eq!(usages, [&usage_36_12()]),
        false => assert!(usages.is_empty(), "{body}"),
    }

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.path, path);
    assert!(sent.body.get("stream_options").is_none());

    let seen = seen(&harness).await;
    let [seen] = seen.as_slice() else {
        panic!("{seen:?}")
    };
    assert_eq!(seen.deployment.as_deref(), Some(id));
    assert_eq!(seen.status, 200);
    assert!(seen.first_byte);
    let usage = seen
        .usage
        .expect("the usage is recorded even when the client did not ask for it");
    assert_eq!(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens
        ),
        (10, 12, 20, 6)
    );
    assert!(
        (seen.cost.unwrap() - expected_cost()).abs() < 1e-9,
        "{seen:?}"
    );
}

#[rstest]
#[tokio::test]
async fn an_anthropic_stream_gets_stream_true() {
    let harness = harness().await;
    let mut body = request("claude", false);
    body["max_tokens"] = json!(8);
    send(&harness.app, body).await;
    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(
        sent.body,
        json!({"model": "claude-x", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "max_tokens": 8, "stream": true})
    );
}

#[rstest]
#[tokio::test]
async fn an_error_before_the_first_chunk_falls_over_to_another_deployment() {
    let harness = harness().await;
    // Whichever deployment the router picks first, the stream comes from the healthy one.
    let (status, _, body) = send(&harness.app, request("fallback", true)).await;
    assert_eq!(status, 200, "{body}");
    let (frames, done) = frames(&body);
    assert!(done);
    assert_eq!(frames[0]["choices"][0]["delta"]["content"], "Hello");
    let seen = seen(&harness).await;
    assert_eq!(seen[0].deployment.as_deref(), Some("bedrock-3"));
    assert_eq!(seen[0].status, 200);
}

#[rstest]
#[tokio::test]
async fn an_upstream_error_status_is_an_openai_error_response() {
    let harness = harness().await;
    let (status, content_type, body) = send(&harness.app, request("overloaded", false)).await;
    assert_eq!(status, 529, "{body}");
    assert_eq!(content_type.as_deref(), Some("application/json"));
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["error"]["code"], 529);
    let seen = seen(&harness).await;
    assert_eq!(seen[0].status, 529);
    assert_eq!(seen[0].usage, None);
}

#[rstest]
#[tokio::test]
async fn an_error_mid_stream_ends_with_the_openai_error_frame() {
    let harness = harness().await;
    let (status, _, body) = send(&harness.app, request("bedrock-cut", true)).await;
    assert_eq!(status, 200, "{body}");
    let (frames, done) = frames(&body);
    assert!(!done, "{body}");
    assert_eq!(frames[0]["choices"][0]["delta"]["content"], "Hel");
    let error = &frames.last().unwrap()["error"];
    assert_eq!(error["code"], 429, "{body}");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("Too many requests"),
        "{body}"
    );
    let seen = seen(&harness).await;
    assert_eq!(seen[0].usage, None);
}
