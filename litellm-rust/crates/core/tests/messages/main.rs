use std::{sync::Arc, time::Duration};

use litellm_core::messages::{
    Error,
    route::{LocalMessagesHost, MessagesCall, MessagesOutput, messages_machine},
    types::MessagesShaping,
};
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use rstest::fixture;
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

#[path = "../support/mod.rs"]
mod support;
use support::*;

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
        model: MODEL.into(),
        body: object(json!({
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

async fn run_with(
    secrets: Arc<RecordingSecrets>,
    call: MessagesCall,
) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(messages_machine(secrets), &LocalMessagesHost::new(call)).await
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
