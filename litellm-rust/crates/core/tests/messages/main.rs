use std::{sync::Arc, time::Duration};

use litellm_core::messages::{
    Error, MessagesCall, MessagesShaping,
    route::{LocalMessagesHost, MessagesMachine, MessagesOutput, messages_machine},
};
use litellm_http::{HttpSettings, Resolution};
use litellm_secrets::source::SecretSource;
use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
};
use rstest::fixture;
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

#[path = "../support/mod.rs"]
mod support;
use support::*;

mod hooks;
mod host;
mod request;
mod response;
mod secrets;
mod stream;

const MODEL: &str = "claude-sonnet-4-5";

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(map) = value else {
        panic!("expected a json object, got {value}");
    };
    map
}

fn body(value: Value) -> AnthropicMessagesRequest {
    serde_json::from_value(value).unwrap()
}

fn with_fields(call: MessagesCall, fields: Value) -> MessagesCall {
    let current = object(serde_json::to_value(&call.body).unwrap());
    MessagesCall {
        body: body(Value::Object(
            current.into_iter().chain(object(fields)).collect(),
        )),
        ..call
    }
}

fn with_model(call: MessagesCall, model: &str) -> MessagesCall {
    with_fields(call, json!({"model": model}))
}

fn message_body() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": MODEL,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2}
    })
}

fn message_response() -> ResponseTemplate {
    json_response(message_body())
}

/// A non-streaming call with nothing that would authenticate or route it, so each test
/// states the provider, credentials, and base it depends on.
#[fixture]
fn call() -> MessagesCall {
    MessagesCall {
        body: body(json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        })),
        api_key: None,
        api_base: None,
        custom_llm_provider: Some("anthropic".into()),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(5)),
        shaping: MessagesShaping::default(),
    }
}

fn headers<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<Map<String, Value>> {
    Some(
        pairs
            .into_iter()
            .map(|(name, value)| (name.to_string(), Value::from(value)))
            .collect(),
    )
}

fn machine(secrets: Arc<dyn SecretSource>) -> MessagesMachine {
    messages_machine(&support::resources(), &http_config(), secrets)
        .expect("default HTTP settings build a client")
}

async fn run_with(
    secrets: Arc<RecordingSecrets>,
    call: MessagesCall,
) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(machine(secrets), &LocalMessagesHost::new(call)).await
}

/// Runs the route with a secret source that knows nothing, so no environment leaks in.
async fn run(call: MessagesCall) -> Result<MessagesOutput, Error> {
    run_with(Arc::new(RecordingSecrets::empty()), call).await
}

async fn run_message(call: MessagesCall) -> AnthropicMessagesResponse {
    match run(call).await.expect("messages call succeeds") {
        MessagesOutput::Message(message) => *message,
        MessagesOutput::Streamed => panic!("a non-streaming call returned a stream"),
    }
}
