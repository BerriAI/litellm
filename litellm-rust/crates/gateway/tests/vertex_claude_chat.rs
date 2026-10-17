//! Claude chat completions on Vertex AI through the gateway (`vertex_ai/claude-*`, a service
//! account traded for an access token): the `rawPredict` request Vertex gets, the OpenAI answer
//! the client gets, and the usage and cost the request is recorded with.

use std::{
    io::Write,
    sync::{Arc, Mutex, Once},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header::CONTENT_TYPE},
    response::IntoResponse,
};
use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_gateway_hooks::{Facts, Observer, Record};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

/// Input and output prices of every deployment.
const RATES: (f64, f64) = (1.0, 2.0);

const ACCESS_TOKEN: &str = "ya29.test-token";
const PROJECT: &str = "test-project";
const LOCATION: &str = "us-east5";
const MODEL: &str = "claude-sonnet-4-5@20250929";

/// A throwaway RSA key the fake service account signs its token request with.
const TEST_PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC95ihJQlfJuaQe
ZCwF+EFy/Y5vk7Lx6Nm74He9Oes+6ZbgA2LVdcoFJrNEYHd8S0CCv7cqEp4dpIoR
B4HAmQAQF92ZC/S2lcgLAezWQU21TSjDLngdVbn+Jr4syzOwVw/NUigydtQ66Iqv
1FpXs/4gQXMgs8FW5XgOhgfKhyMPf71A2qiRaA0vwsrfsmKOY0QVVooXPRfpLIvZ
VEOV89ZMQErnYan+m8OADoZ5NJ/Ij6QKb2HkQJNhQC2Zq2T1Y9VLyxY6pXURdVqc
XJgFT6YPGjLABLX/oId4g+Fs2YI5mM+g72pYEntnIaE5yicGKBdjyqeVfKZ6/7jg
g8evHi6vAgMBAAECggEAGdJjlP6b8Fa5bdaCM/ebcrbuuNZVJVbb0JPHxGfNSLs7
pE9hj5QaOdQW2Uviw3h6F61ZCzQH4xD+Iy2po5ZKb2XHYKnDB1bboj+LRGER337T
9aJqe9at2VTMVEv3Rdm40NsEk0QcPLxlK16NQFK90gYEUSSQPDAswJDSG2R/zHn+
vADI907mW/goEJHeLn8PWGlNlSiR6x+5JJtq+GXCzUzVvJYQSCLGxCSl2x2H+0g7
NhFI0zPpdzNmO/h+yhzaFb6Rp5U8+ZsnZ3qYjQ/03gw1myTDKJt1YaO9JvArnNYX
hcJQQ8Rt0bHhcrZA16bBOpqZlo5pKCicwI/netgN8QKBgQDcFz7AzdJ26sMSV32V
rwrMgIoggt8qDjO1ARwqW35A1TIge0FoW4M4KpsXQGGfT341uU1esXEcyZ/1L/5X
3ql2gX4DbOYLZLWYzZGR2hq33oi8HkhN98QrEwL9emSH8NqYX3Xxja3PrmCrSYJe
Zbnd9TIm2XkxyMoyXJu6M/QvnwKBgQDc4dzqTbxoGEGa5MuJoGmMwPnqgdG9UM5J
eExVnh7osxc2sOdsiPeRjjQTxs9v2kJwctC359OJoo9yGaaJeSghU4LEWJo1sqnA
fzSCLammYvtVAtniyNv5Mxk/6Uimi4NNDKaAKB+m4K2uSn3U9AmY7KPYMGaSbS9W
XSnobjxm8QKBgC8bPpAvvWs8ZhIn7bY659nLbUT2HeO3dHO6UBf0yzn/J6JyHxbB
93zvCZDZc8uQTRgcmCW7XtVlhjoJUqvl+Wlm39zF0xr/LCsPXKfWAb/2/lcdOCaP
8Emz4QD10EyUTYUtcWYJB/mafhBLRH8F0Nlj4J8WDu2L51MOJTqeYhZLAoGAWffN
icocAbJPlo22sdoa4+/+W5yBF8GAJMDRJtZ+9H1t6SLpQHYRkMIBSETkXUTjZvX9
Ocs9iIQkNW9pO/mTdO+VBfCo71JUfknR02xR+6m5gYjlws/ZeYlssXGN2/hbhNiw
QOcW7Vv6olFJK6Iy/oz0t6wPO3kpnN3Zogi0paECgYEAwo44M1DdYCtV0snhmYM9
5u0mPfYt5P2SVLXyUbr+vFTfrTL/WKnXIJgbsnj3Gvf+GIZv9tKcXhSNmEHQCYX4
X3w9iTPddCHuvZ1fpufi2TyArJh0OkoNtLXJHTKrHjf2N+61AQzFiv5WieJrdE+H
qr32PTUuVGPyO9LyTY4/RL0=
-----END PRIVATE KEY-----
";

/// The gateway reads its cost map from this variable; every test sets it before building a
/// gateway, so no test reads the environment while it changes.
fn use_local_cost_map() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let map = json!({
            format!("vertex_ai/{MODEL}"): {
                "mode": "chat",
                "litellm_provider": "vertex_ai-anthropic_models",
                "max_output_tokens": 64000,
                "supports_function_calling": true,
                "supports_tool_choice": true,
                "supports_reasoning": true,
            },
        });
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(map.to_string().as_bytes()).unwrap();
        let (_, path) = file.keep().unwrap();
        // SAFETY: set once, before any test builds a gateway or spawns a thread that reads the
        // environment.
        unsafe { std::env::set_var("LITELLM_MODEL_COST_MAP_PATH", path) };
    });
}

fn message(content: Value, stop_reason: &str) -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    })
}

fn text_answer(text: &str) -> Value {
    message(json!([{"type": "text", "text": text}]), "end_turn")
}

fn tool_answer() -> Value {
    message(
        json!([{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]),
        "tool_use",
    )
}

fn thinking_answer() -> Value {
    message(
        json!([
            {"type": "thinking", "thinking": "The user says hi.", "signature": "sig-1"},
            {"type": "text", "text": "Hello"}
        ]),
        "end_turn",
    )
}

fn sse_answer() -> String {
    [
        json!({"type": "message_start", "message": {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 12}}),
        json!({"type": "message_stop"}),
    ]
    .iter()
    .map(|event| format!("event: {}\ndata: {event}\n\n", event["type"].as_str().unwrap()))
    .collect()
}

#[derive(Clone, Debug)]
struct Sent {
    path: String,
    query: Option<String>,
    authorization: Option<String>,
    api_key: Option<String>,
    anthropic_version: Option<String>,
    anthropic_beta: Option<String>,
    body: Value,
}

type Calls = Arc<Mutex<Vec<Sent>>>;

fn has_tool_result(body: &Value) -> bool {
    body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["content"]
                .as_array()
                .is_some_and(|blocks| blocks.iter().any(|block| block["type"] == "tool_result"))
        })
    })
}

/// Google's token endpoint and Vertex's `rawPredict` and `streamRawPredict`, answering by what
/// the body asks.
async fn upstream(calls: Calls) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let calls = calls.clone();
        async move {
            let (parts, body) = request.into_parts();
            let header = |name: &str| {
                parts
                    .headers
                    .get(name)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned)
            };
            let bytes = to_bytes(body, 1 << 20).await.unwrap();
            let path = parts.uri.path().to_owned();
            if path == "/token" {
                return (
                    [(CONTENT_TYPE, "application/json")],
                    json!({"access_token": ACCESS_TOKEN, "expires_in": 3600, "token_type": "Bearer"})
                        .to_string(),
                )
                    .into_response();
            }
            let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            calls.lock().unwrap().push(Sent {
                path: path.clone(),
                query: parts.uri.query().map(str::to_owned),
                authorization: header("authorization"),
                api_key: header("x-api-key"),
                anthropic_version: header("anthropic-version"),
                anthropic_beta: header("anthropic-beta"),
                body: body.clone(),
            });
            match path.ends_with(":streamRawPredict") {
                true => ([(CONTENT_TYPE, "text/event-stream")], sse_answer()).into_response(),
                false => {
                    let answer = match (has_tool_result(&body), body.get("tools"), body.get("thinking")) {
                        (true, _, _) => text_answer("It is sunny in Paris"),
                        (false, Some(_), _) => tool_answer(),
                        (false, None, Some(_)) => thinking_answer(),
                        (false, None, None) => text_answer("Hello world"),
                    };
                    (
                        StatusCode::OK,
                        [(CONTENT_TYPE, "application/json")],
                        answer.to_string(),
                    )
                        .into_response()
                }
            }
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

#[derive(Debug)]
struct Seen {
    deployment: Option<String>,
    usage: Option<litellm_gateway_hooks::Usage>,
    cost: Option<f64>,
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
                usage: record.outcome.usage,
                cost: record.outcome.cost,
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

fn service_account(upstream: &str) -> String {
    json!({
        "type": "service_account",
        "project_id": "credentials-project",
        "private_key_id": "test-key-id",
        "private_key": TEST_PRIVATE_KEY,
        "client_email": "gateway-test@credentials-project.iam.gserviceaccount.com",
        "client_id": "1",
        "token_uri": format!("{upstream}/token"),
    })
    .to_string()
}

async fn harness() -> Harness {
    use_local_cost_map();
    let calls = Calls::default();
    let upstream = upstream(calls.clone()).await;
    let (input, output) = RATES;
    let prices =
        format!("      input_cost_per_token: {input}\n      output_cost_per_token: {output}\n");
    let credentials = service_account(&upstream);
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: vertex_ai/{MODEL}
      api_base: {upstream}
      vertex_project: {PROJECT}
      vertex_location: {LOCATION}
      vertex_credentials: '{credentials}'
{prices}    model_info:
      id: claude-1
  - model_name: claude-global
    litellm_params:
      model: vertex_ai/{MODEL}
      api_base: {upstream}
      vertex_project: {PROJECT}
      vertex_location: global
      vertex_credentials: '{credentials}'
{prices}    model_info:
      id: claude-global-1
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

async fn send(app: &Router, body: Value) -> (u16, String) {
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-master")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

async fn seen(harness: &Harness) -> Vec<Seen> {
    litellm_gateway::hooks::close(&harness.inference).await;
    std::mem::take(&mut *harness.seen.0.lock().unwrap())
}

fn cost(prompt: u64, completion: u64) -> f64 {
    let (input, output) = RATES;
    prompt as f64 * input + completion as f64 * output
}

fn hi(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "hi"}
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    })
}

fn sent_body() -> Value {
    json!({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "temperature": 0.2,
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "Be brief"}],
    })
}

fn raw_predict_path(location: &str, method: &str) -> String {
    format!(
        "/v1/projects/{PROJECT}/locations/{location}/publishers/anthropic/models/{MODEL}:{method}"
    )
}

#[rstest]
#[case::regional("claude", "claude-1", LOCATION)]
#[case::global("claude-global", "claude-global-1", "global")]
#[tokio::test]
async fn a_chat_completion_calls_raw_predict_and_records_its_cost(
    #[case] model: &str,
    #[case] id: &str,
    #[case] location: &str,
) {
    let harness = harness().await;
    let (status, body) = send(&harness.app, hi(model)).await;
    assert_eq!(status, 200, "{body}");
    let answer: Value = serde_json::from_str(&body).unwrap();
    // `VertexAIAnthropicConfig.transform_response` names the requested model.
    assert_eq!(answer["model"], MODEL);
    assert_eq!(answer["choices"][0]["message"]["role"], "assistant");
    assert_eq!(answer["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");
    assert_eq!(answer["usage"]["prompt_tokens"], 10);
    assert_eq!(answer["usage"]["completion_tokens"], 4);
    assert_eq!(answer["usage"]["total_tokens"], 14);

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.path, raw_predict_path(location, "rawPredict"));
    assert_eq!(sent.query, None);
    assert_eq!(
        sent.authorization.as_deref(),
        Some("Bearer ya29.test-token")
    );
    assert_eq!(sent.api_key, None);
    assert_eq!(sent.anthropic_version.as_deref(), Some("2023-06-01"));
    assert_eq!(sent.anthropic_beta, None);
    assert_eq!(sent.body, sent_body());

    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].deployment.as_deref(), Some(id));
    let usage = seen[0].usage.unwrap();
    assert_eq!((usage.prompt_tokens(), usage.output_tokens), (10, 4));
    assert_eq!(seen[0].cost, Some(cost(10, 4)));
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

#[rstest]
#[case::regional("claude", "claude-1", LOCATION)]
#[case::global("claude-global", "claude-global-1", "global")]
#[tokio::test]
async fn a_stream_calls_stream_raw_predict_and_records_its_usage(
    #[case] model: &str,
    #[case] id: &str,
    #[case] location: &str,
) {
    let harness = harness().await;
    let mut request = hi(model);
    request["stream"] = json!(true);
    request["stream_options"] = json!({"include_usage": true});
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 200, "{body}");

    let (frames, done) = frames(&body);
    assert!(done, "{body}");
    assert!(
        frames
            .iter()
            .all(|frame| frame["object"] == "chat.completion.chunk" && frame["model"] == model),
        "{body}"
    );
    let content: String = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Hello world", "{body}");
    let finishes: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame["choices"][0].get("finish_reason"))
        .filter(|reason| !reason.is_null())
        .collect();
    assert_eq!(finishes, [&json!("stop")], "{body}");
    let usages: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame.get("usage"))
        .filter(|usage| !usage.is_null())
        .collect();
    assert_eq!(usages.len(), 1, "{body}");
    assert_eq!(usages[0]["prompt_tokens"], 10);
    assert_eq!(usages[0]["completion_tokens"], 12);

    let sent = harness.calls.lock().unwrap()[0].clone();
    assert_eq!(sent.path, raw_predict_path(location, "streamRawPredict"));
    // A custom `api_base` asks for SSE, as `_check_custom_proxy` does.
    assert_eq!(sent.query.as_deref(), Some("alt=sse"));
    assert_eq!(
        sent.authorization.as_deref(),
        Some("Bearer ya29.test-token")
    );
    let mut expected = sent_body();
    expected["stream"] = json!(true);
    assert_eq!(sent.body, expected);

    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].deployment.as_deref(), Some(id));
    let usage = seen[0].usage.unwrap();
    assert_eq!((usage.prompt_tokens(), usage.output_tokens), (10, 12));
    assert_eq!(seen[0].cost, Some(cost(10, 12)));
}

#[tokio::test]
async fn a_stream_without_include_usage_still_reports_usage_to_the_hooks() {
    let harness = harness().await;
    let mut request = hi("claude");
    request["stream"] = json!(true);
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 200, "{body}");
    let (frames, done) = frames(&body);
    assert!(done, "{body}");
    assert!(
        frames
            .iter()
            .all(|frame| frame.get("usage").is_none_or(Value::is_null)),
        "{body}"
    );
    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert_eq!(seen[0].cost, Some(cost(10, 12)));
}

#[tokio::test]
async fn a_tool_call_round_trips_through_tool_use_and_tool_result() {
    let harness = harness().await;
    let tools = json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}]);
    let question = json!({"role": "user", "content": "Weather in Paris?"});
    let (status, body) = send(
        &harness.app,
        json!({"model": "claude", "messages": [question], "tools": tools, "tool_choice": "auto", "max_tokens": 64}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let answer: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(answer["choices"][0]["finish_reason"], "tool_calls");
    let call = &answer["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["id"], "toolu_1");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"city\": \"Paris\"}");

    let first = harness.calls.lock().unwrap()[0].body.clone();
    assert_eq!(
        first,
        json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}],
            "tools": [{
                "name": "get_weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                "type": "custom",
                "description": "Weather for a city"
            }],
            "tool_choice": {"type": "auto"},
            "max_tokens": 64,
            "anthropic_version": "vertex-2023-10-16",
        })
    );

    let (status, body) = send(
        &harness.app,
        json!({"model": "claude", "tools": tools, "max_tokens": 64, "messages": [
            question,
            {"role": "assistant", "content": null, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}
        ]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let answer: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        answer["choices"][0]["message"]["content"],
        "It is sunny in Paris"
    );
    let second = harness.calls.lock().unwrap()[1].body.clone();
    assert_eq!(
        second["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]}
        ])
    );

    let seen = seen(&harness).await;
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert!(seen.iter().all(|seen| seen.cost == Some(cost(10, 4))));
}

#[tokio::test]
async fn reasoning_effort_becomes_a_thinking_budget_and_comes_back_as_reasoning() {
    let harness = harness().await;
    let (status, body) = send(
        &harness.app,
        json!({"model": "claude", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let answer: Value = serde_json::from_str(&body).unwrap();
    let message = &answer["choices"][0]["message"];
    assert_eq!(message["content"], "Hello");
    assert_eq!(message["reasoning_content"], "The user says hi.");
    assert_eq!(
        message["thinking_blocks"],
        json!([{"type": "thinking", "thinking": "The user says hi.", "signature": "sig-1"}])
    );

    let sent = harness.calls.lock().unwrap()[0].body.clone();
    assert_eq!(
        sent,
        json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "max_tokens": 1024 + 4096,
            "anthropic_version": "vertex-2023-10-16",
        })
    );
    let seen = seen(&harness).await;
    assert_eq!(seen[0].cost, Some(cost(10, 4)));
}

#[tokio::test]
async fn an_anthropic_api_only_param_is_refused() {
    let harness = harness().await;
    let mut request = hi("claude");
    request["service_tier"] = json!("auto");
    let (status, body) = send(&harness.app, request).await;
    assert_eq!(status, 400, "{body}");
    assert!(harness.calls.lock().unwrap().is_empty());
}
