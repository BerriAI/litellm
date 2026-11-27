//! Every native Messages provider runs the same cases through the public route, and what goes
//! on the wire plus what the caller gets back is compared with a golden file recorded from main
//! under `tests/messages/conformance/<provider>/<case>.json`. Goldens are never hand-edited:
//! run with `LITELLM_UPDATE_GOLDENS=1` to rewrite them, and only for an intended change.

use std::path::PathBuf;

use base64::{Engine, engine::general_purpose::STANDARD};
use bytes::BytesMut;
use futures_util::TryStreamExt;
use litellm_framer::aws_event_stream::{AwsEventStreamCodec, Header, HeaderValue, Message};
use litellm_inference_messages::MessagesCallResponse;
use litellm_llms::base_llm::messages::context::{
    BedrockCapabilities, MessagesModelCapabilities, NativeOutputSupport,
};
use regex::Regex;
use rstest::rstest;
use tokio_util::codec::Encoder;
use wiremock::{MockServer, Request};

use super::*;

const UPDATE_GOLDENS: &str = "LITELLM_UPDATE_GOLDENS";
const API_KEY: &str = "sk-conformance-key";
const AWS_ACCESS_KEY_ID: &str = "AKIACONFORMANCE";
const AWS_SECRET_ACCESS_KEY: &str = "conformance-aws-secret";
const GOOGLE_TOKEN: &str = "conformance-google-token";
const SECRET_HEADERS: [&str; 4] = [
    "api-key",
    "authorization",
    "x-api-key",
    "x-amz-security-token",
];
const UNSTABLE_HEADERS: [&str; 3] = ["date", "x-amz-date", "user-agent"];

const STREAM_EVENTS: [&str; 6] = [
    r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1}}}"#,
    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
    r#"{"type":"content_block_stop","index":0}"#,
    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}"#,
    r#"{"type":"message_stop"}"#,
];

struct Provider {
    route: litellm_inference_messages::MessagesRoute,
    call: MessagesCall,
    streams_aws_events: bool,
}

fn provider(name: &str, upstream: &MockServer) -> Provider {
    let base = upstream.uri();
    let keyed = |custom_llm_provider: Option<&str>, model: &str, litellm_params: Value| Provider {
        route: messages_route(Arc::new(RecordingSecrets::empty())),
        call: MessagesCall {
            api_key: Some(API_KEY.into()),
            api_base: Some(base.clone()),
            custom_llm_provider: custom_llm_provider.map(Into::into),
            litellm_params: match litellm_params {
                Value::Object(fields) if !fields.is_empty() => {
                    serde_json::from_value(Value::Object(
                        fields
                            .into_iter()
                            .chain([("model".into(), json!(model))])
                            .collect(),
                    ))
                    .unwrap()
                }
                _ => Default::default(),
            },
            ..with_model(call(), model)
        },
        streams_aws_events: false,
    };
    match name {
        "anthropic" => keyed(Some("anthropic"), MODEL, json!({})),
        "azure_ai" => keyed(Some("azure_ai"), MODEL, json!({})),
        "deepseek" => keyed(Some("deepseek"), MODEL, json!({})),
        "edenai" => keyed(Some("edenai"), MODEL, json!({})),
        "minimax" => keyed(Some("minimax"), MODEL, json!({})),
        "tencent" => keyed(Some("tencent"), MODEL, json!({})),
        "openrouter" => {
            let openrouter = keyed(None, "openrouter/anthropic/claude-sonnet-4.5", json!({}));
            Provider {
                call: MessagesCall {
                    api_base: Some(format!("{base}/api/v1")),
                    ..openrouter.call
                },
                ..openrouter
            }
        }
        "bedrock_mantle" => keyed(
            None,
            "bedrock_mantle/anthropic.claude-sonnet-4-5",
            json!({}),
        ),
        "bedrock_mantle_route" => keyed(
            None,
            "bedrock/mantle/anthropic.claude-sonnet-4-5",
            json!({}),
        ),
        "bedrock_claude_platform" => keyed(
            None,
            "bedrock/claude_platform/claude-sonnet-4-5",
            json!({"workspace_id": "conformance-workspace"}),
        ),
        "bedrock_invoke" => {
            let invoke = keyed(
                None,
                "bedrock/invoke/us-west-2/anthropic.claude-sonnet-4-5-20250929-v1:0",
                json!({
                    "aws_access_key_id": AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
                }),
            );
            Provider {
                call: MessagesCall {
                    api_key: None,
                    ..invoke.call
                },
                streams_aws_events: true,
                ..invoke
            }
        }
        "vertex_ai" => Provider {
            route: messages_route_with_auth(
                Arc::new(RecordingSecrets::empty()),
                litellm_auth::AuthServices {
                    vertex_login: litellm_auth_gcp::vertex::VertexLoginService::new(Arc::new(
                        litellm_auth_gcp::google::OnGce {
                            project: "token-project",
                            token: GOOGLE_TOKEN,
                        },
                    )),
                    ..litellm_auth::AuthServices::default()
                },
            ),
            call: MessagesCall {
                api_key: None,
                ..keyed(
                    Some("vertex_ai"),
                    "claude-sonnet-4-5@20250929",
                    json!({"vertex_project": "conformance-project", "vertex_location": "us-east5"}),
                )
                .call
            },
            streams_aws_events: false,
        },
        other => panic!("no conformance setup for provider {other}"),
    }
}

fn case_fields(case: &str) -> (Value, Option<Map<String, Value>>) {
    match case {
        "plain" => (json!({}), None),
        "stream" => (json!({"stream": true}), None),
        "system_tools_tool_choice" => (
            json!({
                "system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
                "tools": [
                    {"name": "get_weather", "description": "Weather for a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
                    {"type": "custom", "name": "get_time", "input_schema": {"type": "object"}}
                ],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "messages": [
                    {"role": "user", "content": "weather in Paris?"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "sunny"}]}
                ]
            }),
            None,
        ),
        "thinking_effort" => (
            json!({
                "max_tokens": 2048,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "output_config": {"effort": "high"}
            }),
            None,
        ),
        "droppable_params" => (
            json!({
                "service_tier": "auto",
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
                "output_format": {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}},
                "metadata": {"user_id": "conformance-user"}
            }),
            None,
        ),
        "anthropic_beta" => (
            json!({}),
            headers([(
                "anthropic-beta",
                "context-1m-2025-08-07,interleaved-thinking-2025-05-14,unknown-beta-2099-01-01",
            )]),
        ),
        "extra_headers" => (
            json!({}),
            headers([
                ("x-custom-header", "custom-value"),
                ("x-request-source", "conformance"),
            ]),
        ),
        "host_edits" => (
            json!({
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                    {"type": "text", "text": "be terse"}
                ],
                "output_format": {"type": "json_schema", "schema": {"type": "object"}},
                "output_config": {"effort": "low", "format": {"type": "replaced"}}
            }),
            headers([("anthropic-version", "2023-06-01")]),
        ),
        "non_object_output_format" => (json!({"output_format": "text"}), None),
        other => panic!("no conformance case {other}"),
    }
}

fn sse_body() -> Vec<u8> {
    STREAM_EVENTS
        .iter()
        .map(|event| {
            let kind = serde_json::from_str::<Value>(event).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_string();
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect::<String>()
        .into_bytes()
}

fn aws_event_stream_body() -> Vec<u8> {
    let mut wire = BytesMut::new();
    for event in STREAM_EVENTS {
        let message = Message::new(format!(r#"{{"bytes": "{}"}}"#, STANDARD.encode(event)))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String("chunk".into()),
            ))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ));
        AwsEventStreamCodec.encode(message, &mut wire).unwrap();
    }
    wire.to_vec()
}

fn upstream_response(stream: bool, streams_aws_events: bool) -> ResponseTemplate {
    match (stream, streams_aws_events) {
        (false, _) => message_response(),
        (true, false) => ResponseTemplate::new(200).set_body_raw(sse_body(), "text/event-stream"),
        (true, true) => ResponseTemplate::new(200).set_body_raw(
            aws_event_stream_body(),
            "application/vnd.amazon.eventstream",
        ),
    }
}

struct Normaliser {
    replacements: Vec<(String, &'static str)>,
    sigv4_date: Regex,
    sigv4_signature: Regex,
}

impl Normaliser {
    fn new(upstream: &MockServer) -> Self {
        let address = upstream.address().to_string();
        Self {
            replacements: vec![
                (upstream.uri(), "<upstream>"),
                (address, "<upstream-address>"),
                (API_KEY.into(), "<api-key>"),
                (AWS_ACCESS_KEY_ID.into(), "<aws-access-key-id>"),
                (GOOGLE_TOKEN.into(), "<google-token>"),
            ],
            sigv4_date: Regex::new(r"/\d{8}/").unwrap(),
            sigv4_signature: Regex::new(r"Signature=[0-9a-f]+").unwrap(),
        }
    }

    fn text(&self, text: &str) -> String {
        self.replacements
            .iter()
            .fold(text.to_string(), |text, (from, to)| text.replace(from, to))
    }

    fn header(&self, name: &str, value: &str) -> String {
        if UNSTABLE_HEADERS.contains(&name) {
            return format!("<{name}>");
        }
        let value = self.text(value);
        if !SECRET_HEADERS.contains(&name) || value.contains('<') {
            return self
                .sigv4_signature
                .replace_all(
                    &self.sigv4_date.replace_all(&value, "/<date>/"),
                    "Signature=<signature>",
                )
                .into_owned();
        }
        "<redacted>".into()
    }

    fn request(&self, request: &Request) -> Value {
        let mut names: Vec<&str> = request.headers.keys().map(|name| name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        let headers: Map<String, Value> = names
            .into_iter()
            .map(|name| {
                let values = request
                    .header_values(name)
                    .into_iter()
                    .map(|value| Value::from(self.header(name, value)))
                    .collect();
                (name.to_string(), Value::Array(values))
            })
            .collect();
        let body = serde_json::from_slice(&request.body)
            .unwrap_or_else(|_| Value::from(request.body_text()));
        json!({
            "method": request.method.as_str(),
            "target": self.text(&request.target()),
            "headers": headers,
            "body": body,
        })
    }
}

async fn outcome(result: Result<MessagesCallResponse, Error>, normaliser: &Normaliser) -> Value {
    let failure = |error: Error| json!({"error": normaliser.text(&litellm_error::Report::new(&error).to_string())});
    match result {
        Ok(MessagesCallResponse::Complete(body)) => json!({"complete": body}),
        Ok(MessagesCallResponse::Stream { chunks, .. }) => {
            match chunks.try_collect::<Vec<_>>().await {
                Ok(chunks) => json!({"stream": String::from_utf8_lossy(&chunks.concat())}),
                Err(error) => failure(error),
            }
        }
        Err(error) => failure(error),
    }
}

fn golden_path(provider: &str, case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/messages/conformance")
        .join(provider)
        .join(format!("{case}.json"))
}

fn compare_with_golden(provider: &str, case: &str, recorded: &Value) {
    let path = golden_path(provider, case);
    let rendered = serde_json::to_string_pretty(recorded).unwrap() + "\n";
    if std::env::var_os(UPDATE_GOLDENS).is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, rendered).unwrap();
        return;
    }
    let golden = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} is unreadable ({error}); record it with {UPDATE_GOLDENS}=1",
            path.display()
        )
    });
    assert_eq!(
        serde_json::from_str::<Value>(&golden).unwrap(),
        *recorded,
        "{provider} {case} differs from {}",
        path.display()
    );
}

#[rstest]
#[tokio::test]
async fn every_provider_sends_and_decodes_each_case_as_main_does(
    #[values(
        "anthropic",
        "azure_ai",
        "bedrock_invoke",
        "bedrock_mantle_route",
        "bedrock_claude_platform",
        "bedrock_mantle",
        "vertex_ai",
        "deepseek",
        "tencent",
        "minimax",
        "openrouter",
        "edenai"
    )]
    provider_name: &str,
    #[values(
        "plain",
        "stream",
        "system_tools_tool_choice",
        "thinking_effort",
        "droppable_params",
        "anthropic_beta",
        "extra_headers",
        "host_edits",
        "non_object_output_format"
    )]
    case: &str,
) {
    let (fields, extra_headers) = case_fields(case);
    let stream = fields["stream"] == json!(true);
    let server = MockServer::start().await;
    let Provider {
        route,
        call,
        streams_aws_events,
    } = provider(provider_name, &server);
    let recorded = exchange(
        &server,
        route,
        MessagesCall {
            extra_headers,
            ..with_fields(call, fields)
        },
        [upstream_response(stream, streams_aws_events)],
    )
    .await;

    compare_with_golden(provider_name, case, &recorded);
}

#[rstest]
#[tokio::test]
async fn a_rejected_native_output_is_sent_again_without_it_as_main_does(
    #[values("bedrock_invoke", "bedrock_mantle_route", "bedrock_mantle")] provider_name: &str,
    #[values("native_output_retry", "native_output_retry_stream")] case: &str,
) {
    let stream = case == "native_output_retry_stream";
    let server = MockServer::start().await;
    let Provider {
        route,
        call,
        streams_aws_events,
    } = provider(provider_name, &server);
    let output_format = json!({
        "output_format": {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}
    });
    let requested = match stream {
        true => with_fields(with_fields(call, output_format), json!({"stream": true})),
        false => with_fields(call, output_format),
    };
    let recorded = exchange(
        &server,
        route,
        MessagesCall {
            shaping: MessagesShaping {
                capabilities: MessagesModelCapabilities {
                    bedrock: BedrockCapabilities {
                        native_output: NativeOutputSupport {
                            enabled: true,
                            ..NativeOutputSupport::default()
                        },
                        ..BedrockCapabilities::default()
                    },
                    ..MessagesModelCapabilities::default()
                },
                ..requested.shaping
            },
            ..requested
        },
        [
            status_response(
                400,
                json!({"message": "output_config.format: Extra inputs are not permitted"}),
            ),
            upstream_response(stream, streams_aws_events),
        ],
    )
    .await;

    compare_with_golden(provider_name, case, &recorded);
}

async fn exchange(
    server: &MockServer,
    route: litellm_inference_messages::MessagesRoute,
    call: MessagesCall,
    responses: impl IntoIterator<Item = ResponseTemplate>,
) -> Value {
    respond_in_order(server, responses).await;
    let normaliser = Normaliser::new(server);
    let result = route.execute(call, &(), None).await;
    let outcome = outcome(result, &normaliser).await;
    let requests: Vec<Value> = received(server)
        .await
        .iter()
        .map(|request| normaliser.request(request))
        .collect();
    json!({"requests": requests, "outcome": outcome})
}
