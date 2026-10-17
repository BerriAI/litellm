//! Chat completions for deployments the gateway translates (Anthropic, Bedrock Converse): the
//! OpenAI request widened to tools, images and the provider's own limits, the provider body
//! the upstream sees, and the OpenAI answer the client gets back, streamed and not.

use std::{
    io::Write,
    sync::{Arc, Mutex, Once},
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    http::{Request, header::CONTENT_TYPE},
    response::IntoResponse,
};
use litellm_config::Config;
use litellm_framer::aws_event_stream::{AwsEventStreamCodec, Header, HeaderValue, Message};
use rstest::rstest;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::codec::Encoder;
use tower::ServiceExt;

const ANTHROPIC_MODEL: &str = "claude-x";
const BEDROCK_MODEL: &str = "anthropic.claude-x-v1:0";
/// A one-pixel PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

/// The gateway reads its cost map from this variable; every test sets it before building a
/// gateway, so no test reads the environment while it changes.
fn use_local_cost_map() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let capabilities = json!({
            "mode": "chat",
            "max_tokens": 8192,
            "max_output_tokens": 8192,
            "supports_tool_choice": true,
            "supports_function_calling": true,
            "supports_vision": true,
            "supports_pdf_input": true,
            "supports_prompt_caching": true,
            "supports_response_schema": true,
        });
        let entry = |provider: &str| {
            let mut entry = capabilities.clone();
            entry["litellm_provider"] = json!(provider);
            entry
        };
        let map = json!({
            ANTHROPIC_MODEL: entry("anthropic"),
            BEDROCK_MODEL: entry("bedrock_converse"),
        });
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(map.to_string().as_bytes()).unwrap();
        let (_, path) = file.keep().unwrap();
        // SAFETY: set once, before any test builds a gateway or spawns a thread that reads the
        // environment.
        unsafe { std::env::set_var("LITELLM_MODEL_COST_MAP_PATH", path) };
    });
}

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

fn anthropic_tool_message() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
        ],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

/// The answer to a `response_format` request, given as the tool it became.
fn anthropic_json_message() -> Value {
    json!({
        "id": "msg_3", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
        "content": [{"type": "tool_use", "id": "toolu_2", "name": "json_tool_call", "input": {"city": "Paris"}}],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

fn anthropic_text_message() -> Value {
    json!({
        "id": "msg_2", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": "Sunny."}],
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 2}
    })
}

fn anthropic_tool_stream() -> String {
    anthropic_events(&[
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": ANTHROPIC_MODEL, "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 5}}),
        json!({"type": "message_stop"}),
    ])
}

fn converse_tool_response() -> Value {
    json!({
        "output": {"message": {"role": "assistant", "content": [
            {"text": "Checking."},
            {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather", "input": {"city": "Paris"}}}
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    })
}

fn converse_text_response() -> Value {
    json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "Sunny."}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}
    })
}

fn bedrock_frame(name: &str, payload: Value, wire: &mut bytes::BytesMut) {
    let message = Message::new(Bytes::from(payload.to_string()))
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String("event".to_owned().into()),
        ))
        .add_header(Header::new(
            ":event-type",
            HeaderValue::String(name.to_owned().into()),
        ));
    AwsEventStreamCodec.encode(message, wire).unwrap();
}

fn converse_tool_stream() -> Bytes {
    let mut wire = bytes::BytesMut::new();
    for (name, payload) in [
        ("messageStart", json!({"role": "assistant"})),
        (
            "contentBlockStart",
            json!({"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{\"city\": "}}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": "\"Paris\"}"}}}),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "tool_use"})),
        (
            "metadata",
            json!({"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 9}}),
        ),
    ] {
        bedrock_frame(name, payload, &mut wire);
    }
    wire.freeze()
}

#[derive(Clone, Debug)]
struct Sent {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Sent {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

type Calls = Arc<Mutex<Vec<Sent>>>;

/// Answers each provider by path: a tool call while the conversation has no tool result yet,
/// the final answer once it does.
async fn upstream(calls: Calls) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(move |request: Request<Body>| {
        let calls = calls.clone();
        async move {
            let (parts, body) = request.into_parts();
            let body: Value =
                serde_json::from_slice(&to_bytes(body, 1 << 22).await.unwrap()).unwrap();
            let path = parts.uri.path().to_owned();
            calls.lock().unwrap().push(Sent {
                path: path.clone(),
                headers: parts
                    .headers
                    .iter()
                    .map(|(name, value)| {
                        (
                            name.to_string(),
                            value.to_str().unwrap_or_default().to_owned(),
                        )
                    })
                    .collect(),
                body: body.clone(),
            });
            let answered =
                body.to_string().contains("tool_result") || body.to_string().contains("toolResult");
            let json = [(CONTENT_TYPE, "application/json")];
            if path.ends_with("/converse-stream") {
                return (
                    [(CONTENT_TYPE, "application/vnd.amazon.eventstream")],
                    converse_tool_stream(),
                )
                    .into_response();
            }
            if path.ends_with("/converse") {
                let answer = match answered {
                    true => converse_text_response(),
                    false => converse_tool_response(),
                };
                return (json, answer.to_string()).into_response();
            }
            if body["stream"] == json!(true) {
                return (
                    [(CONTENT_TYPE, "text/event-stream")],
                    anthropic_tool_stream(),
                )
                    .into_response();
            }
            let answer = match answered {
                _ if body["tools"][0]["name"] == "json_tool_call" => anthropic_json_message(),
                true => anthropic_text_message(),
                false => anthropic_tool_message(),
            };
            (json, answer.to_string()).into_response()
        }
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

struct Harness {
    app: Router,
    calls: Calls,
}

async fn harness() -> Harness {
    use_local_cost_map();
    let calls = Calls::default();
    let upstream = upstream(calls.clone()).await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/{ANTHROPIC_MODEL}
      api_base: {upstream}
      api_key: sk-anthropic
  - model_name: bedrock
    litellm_params:
      model: bedrock/{BEDROCK_MODEL}
      api_base: {upstream}
      api_key: bedrock-token
  - model_name: claude-dropping
    litellm_params:
      model: anthropic/{ANTHROPIC_MODEL}
      api_base: {upstream}
      api_key: sk-anthropic
      drop_params: true
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
"
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    litellm_gateway::hooks::install(&inference, &config, Vec::new(), Vec::new())
        .await
        .unwrap();
    let app = litellm_gateway::router(inference, &config, None, None);
    Harness { app, calls }
}

impl Harness {
    async fn send(&self, body: Value) -> (u16, String) {
        let request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer sk-master")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = self.app.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let body = to_bytes(response.into_body(), 1 << 22).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn json(&self, body: Value) -> Value {
        let (status, text) = self.send(body).await;
        assert_eq!(status, 200, "{text}");
        serde_json::from_str(&text).unwrap()
    }

    fn sent(&self) -> Vec<Sent> {
        self.calls.lock().unwrap().clone()
    }
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "The weather in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }})
}

fn question() -> Value {
    json!({"role": "user", "content": "Weather in Paris?"})
}

/// The conversation after the model asked for the tool and the tool answered.
fn answered(call_id: &str) -> Value {
    json!([
        question(),
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": "sunny, 21C"}
    ])
}

/// The streamed chunks' JSON and whether the stream ended with `[DONE]`.
fn frames(body: &str) -> (Vec<Value>, bool) {
    let mut done = false;
    let frames = body
        .split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter(|data| {
            let last = *data == "[DONE]";
            done |= last;
            !last
        })
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    (frames, done)
}

/// The tool calls the chunks stream, joined: (id, name, arguments) per index.
fn streamed_tool_calls(chunks: &[Value]) -> Vec<(String, String, String)> {
    let mut calls: Vec<(String, String, String)> = Vec::new();
    for chunk in chunks {
        let Some(deltas) = chunk["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for delta in deltas {
            let index = delta["index"].as_u64().unwrap() as usize;
            if calls.len() <= index {
                calls.resize(index + 1, Default::default());
            }
            let call = &mut calls[index];
            if let Some(id) = delta["id"].as_str() {
                call.0 = id.to_owned();
            }
            if let Some(name) = delta["function"]["name"].as_str() {
                call.1 = name.to_owned();
            }
            call.2
                .push_str(delta["function"]["arguments"].as_str().unwrap_or_default());
        }
    }
    calls
}

fn finish_reason(chunks: &[Value]) -> Option<String> {
    chunks
        .iter()
        .find_map(|chunk| chunk["choices"][0]["finish_reason"].as_str())
        .map(str::to_owned)
}

#[tokio::test]
async fn anthropic_tool_round_trip() {
    let harness = harness().await;

    let asked = harness
        .json(json!({
            "model": "claude",
            "messages": [question()],
            "tools": [weather_tool()],
            "tool_choice": "required",
            "max_tokens": 64,
        }))
        .await;
    let message = &asked["choices"][0]["message"];
    assert_eq!(asked["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(message["content"], "Checking.");
    assert_eq!(
        message["tool_calls"],
        json!([{"id": "toolu_1", "type": "function", "index": 1,
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}])
    );

    let done = harness
        .json(json!({
            "model": "claude",
            "messages": answered("toolu_1"),
            "tools": [weather_tool()],
            "max_tokens": 64,
        }))
        .await;
    assert_eq!(done["choices"][0]["message"]["content"], "Sunny.");
    assert_eq!(done["choices"][0]["finish_reason"], "stop");

    let sent = harness.sent();
    assert_eq!(sent[0].path, "/v1/messages");
    assert_eq!(
        sent[0].body,
        json!({
            "model": ANTHROPIC_MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}],
            "max_tokens": 64,
            "tools": [{"name": "get_weather", "type": "custom", "description": "The weather in a city",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
            "tool_choice": {"type": "any"}
        })
    );
    assert_eq!(
        sent[1].body["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny, 21C"}]}
        ])
    );
}

#[tokio::test]
async fn bedrock_tool_round_trip() {
    let harness = harness().await;

    let asked = harness
        .json(json!({
            "model": "bedrock",
            "messages": [question()],
            "tools": [weather_tool()],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 64,
        }))
        .await;
    assert_eq!(asked["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        asked["choices"][0]["message"]["tool_calls"],
        json!([{"id": "tooluse_1", "type": "function", "index": 1,
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}])
    );

    let done = harness
        .json(json!({
            "model": "bedrock",
            "messages": answered("tooluse_1"),
            "tools": [weather_tool()],
            "max_tokens": 64,
        }))
        .await;
    assert_eq!(done["choices"][0]["message"]["content"], "Sunny.");
    assert_eq!(done["choices"][0]["finish_reason"], "stop");

    let sent = harness.sent();
    assert_eq!(sent[0].path, format!("/model/{BEDROCK_MODEL}/converse"));
    assert_eq!(
        sent[0].body["toolConfig"],
        json!({
            "tools": [{"toolSpec": {"name": "get_weather", "description": "The weather in a city",
                "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}}],
            "toolChoice": {"tool": {"name": "get_weather"}}
        })
    );
    assert_eq!(
        sent[1].body["messages"],
        json!([
            {"role": "user", "content": [{"text": "Weather in Paris?"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather", "input": {"city": "Paris"}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "tooluse_1", "content": [{"text": "sunny, 21C"}]}}]}
        ])
    );
}

#[rstest]
#[case::anthropic("claude", "toolu_1")]
#[case::bedrock("bedrock", "tooluse_1")]
#[tokio::test]
async fn a_streamed_tool_call_arrives_as_openai_tool_call_chunks(
    #[case] model: &str,
    #[case] id: &str,
) {
    let harness = harness().await;

    let (status, body) = harness
        .send(json!({
            "model": model,
            "messages": [question()],
            "tools": [weather_tool()],
            "max_tokens": 64,
            "stream": true,
        }))
        .await;

    assert_eq!(status, 200, "{body}");
    let (chunks, done) = frames(&body);
    assert!(done, "{body}");
    assert_eq!(
        streamed_tool_calls(&chunks),
        [(
            id.to_owned(),
            "get_weather".to_owned(),
            "{\"city\": \"Paris\"}".to_owned()
        )]
    );
    assert_eq!(finish_reason(&chunks).as_deref(), Some("tool_calls"));
    let sent = &harness.sent()[0];
    let tools = match model {
        "claude" => sent.body["tools"][0]["name"].clone(),
        _ => sent.body["toolConfig"]["tools"][0]["toolSpec"]["name"].clone(),
    };
    assert_eq!(tools, "get_weather");
}

#[rstest]
#[case::max_tokens("max_tokens")]
#[case::max_completion_tokens("max_completion_tokens")]
#[tokio::test]
async fn bedrock_takes_the_token_limit_in_its_inference_config(#[case] param: &str) {
    let harness = harness().await;

    harness
        .json(json!({
            "model": "bedrock",
            "messages": [{"role": "system", "content": "Be brief."}, question()],
            param: 32,
            "temperature": 0.5,
            "stop": ["END"],
        }))
        .await;

    let sent = &harness.sent()[0];
    assert_eq!(
        sent.body["inferenceConfig"],
        json!({"maxTokens": 32, "temperature": 0.5, "stopSequences": ["END"]})
    );
    assert_eq!(sent.body["system"], json!([{"text": "Be brief."}]));
}

#[tokio::test]
async fn anthropic_gets_the_model_default_max_tokens_when_the_caller_names_none() {
    let harness = harness().await;

    harness
        .json(json!({"model": "claude", "messages": [question()]}))
        .await;

    assert_eq!(harness.sent()[0].body["max_tokens"], 8192);
}

#[tokio::test]
async fn an_image_data_url_reaches_each_provider_inline() {
    let harness = harness().await;
    let messages = json!([{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG}")}}
    ]}]);

    harness
        .json(json!({"model": "claude", "messages": messages, "max_tokens": 16}))
        .await;
    harness
        .json(json!({"model": "bedrock", "messages": messages, "max_tokens": 16}))
        .await;

    let sent = harness.sent();
    assert_eq!(
        sent[0].body["messages"][0]["content"][1],
        json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}})
    );
    assert_eq!(
        sent[1].body["messages"][0]["content"][1],
        json!({"image": {"format": "png", "source": {"bytes": PNG}}})
    );
}

#[tokio::test]
async fn anthropic_response_format_comes_back_as_the_answer() {
    let harness = harness().await;
    let schema =
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]});

    let answer = harness
        .json(json!({
            "model": "claude",
            "messages": [question()],
            "max_tokens": 64,
            "response_format": {"type": "json_schema", "json_schema": {"name": "place", "schema": schema}},
            "user": "end-user-1",
        }))
        .await;

    assert_eq!(
        answer["choices"][0]["message"]["content"],
        "{\"city\": \"Paris\"}"
    );
    assert_eq!(answer["choices"][0]["message"]["tool_calls"], Value::Null);
    assert_eq!(answer["choices"][0]["finish_reason"], "stop");
    let sent = &harness.sent()[0];
    assert_eq!(sent.body["metadata"], json!({"user_id": "end-user-1"}));
    assert_eq!(sent.body["tools"][0]["name"], "json_tool_call");
    assert_eq!(sent.body["tools"][0]["input_schema"], schema);
    assert_eq!(
        sent.body["tool_choice"],
        json!({"name": "json_tool_call", "type": "tool"})
    );
}

#[rstest]
#[case::anthropic("claude", json!({"logprobs": true}))]
#[case::bedrock("bedrock", json!({"tool_choice": "none", "tools": [weather_tool()]}))]
#[case::n_choices("claude", json!({"n": 2}))]
#[tokio::test]
async fn an_unsupported_param_is_refused_and_dropped_with_drop_params(
    #[case] model: &str,
    #[case] extra: Value,
) {
    let harness = harness().await;
    let mut body = json!({"model": model, "messages": [question()], "max_tokens": 16});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());

    let (status, refused) = harness.send(body.clone()).await;
    assert_eq!(status, 400, "{refused}");
    assert!(
        harness.sent().is_empty(),
        "a refused request never goes out"
    );

    body["drop_params"] = json!(true);
    harness.json(body).await;
    let sent = &harness.sent()[0];
    for name in extra.as_object().unwrap().keys() {
        if name == "tools" {
            continue;
        }
        assert!(
            sent.body.get(name).is_none(),
            "{name} reached {}",
            sent.body
        );
    }
    assert!(
        sent.body
            .get("toolConfig")
            .and_then(|config| config.get("toolChoice"))
            .is_none()
    );
    assert!(sent.body.get("drop_params").is_none());
}

#[tokio::test]
async fn a_deployment_that_drops_params_drops_them_without_the_request_asking() {
    let harness = harness().await;

    harness
        .json(json!({
            "model": "claude-dropping",
            "messages": [question()],
            "max_tokens": 16,
            "logprobs": true,
        }))
        .await;

    assert!(harness.sent()[0].body.get("logprobs").is_none());
}

#[tokio::test]
async fn a_cache_control_part_reaches_anthropic_and_bedrock() {
    let harness = harness().await;
    let messages = json!([
        {"role": "system", "content": [{"type": "text", "text": "Long instructions.", "cache_control": {"type": "ephemeral"}}]},
        question()
    ]);

    harness
        .json(json!({"model": "claude", "messages": messages, "max_tokens": 16}))
        .await;
    harness
        .json(json!({"model": "bedrock", "messages": messages, "max_tokens": 16}))
        .await;

    let sent = harness.sent();
    assert_eq!(
        sent[0].body["system"],
        json!([{"type": "text", "text": "Long instructions.", "cache_control": {"type": "ephemeral"}}])
    );
    assert_eq!(
        sent[1].body["system"],
        json!([{"text": "Long instructions."}, {"cachePoint": {"type": "default"}}])
    );
    assert!(sent[0].header("x-api-key").is_some());
}
