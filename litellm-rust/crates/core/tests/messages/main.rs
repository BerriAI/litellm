use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_core::messages::{
    Error, MessagesCall, MessagesShaping,
    route::{Messages, MessagesMachine, MessagesOutput},
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

fn machine(secrets: Arc<dyn SecretSource>) -> impl FnOnce(MessagesCall) -> MessagesMachine {
    client_with_secrets(secrets)
        .messages_machine()
        .expect("default HTTP settings build a client")
}

async fn run_with(
    secrets: Arc<RecordingSecrets>,
    call: MessagesCall,
) -> Result<MessagesOutput, Error> {
    let host = LocalMessagesHost::new(call);
    litellm_host::in_process::run_hosted(machine(secrets)(host.request()?), host.runtime()).await
}

/// Runs the route with a secret source that knows nothing, so no environment leaks in.
async fn run(call: MessagesCall) -> Result<MessagesOutput, Error> {
    run_with(Arc::new(RecordingSecrets::empty()), call).await
}

async fn run_message(call: MessagesCall) -> AnthropicMessagesResponse {
    match run(call).await.expect("messages call succeeds") {
        MessagesOutput::Complete(message) => *message,
        MessagesOutput::StreamEnded | MessagesOutput::Detached => {
            panic!("a non-streaming call returned a stream")
        }
    }
}

struct LocalMessagesHost {
    call: Mutex<Option<MessagesCall>>,
}

impl LocalMessagesHost {
    fn new(call: MessagesCall) -> Self {
        Self {
            call: Mutex::new(Some(call)),
        }
    }
}

impl LocalMessagesHost {
    pub fn request(&self) -> Result<MessagesCall, Error> {
        self.call
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .ok_or_else(|| Error::InvalidRequest("messages request was already projected".into()))
    }
    pub fn runtime(&self) -> litellm_host::in_process::Host<'_, (), Self, ()> {
        litellm_host::in_process::Host {
            services: &(),
            hooks: self,
            stream: &(),
            observer: Some(self),
        }
    }
}

impl litellm_host::lifecycle::CallObserver for LocalMessagesHost {
    fn observe(&self, _: litellm_host::event::CallEvent) {}
}
impl litellm_host::hooks::RouteHooks<<Messages as litellm_host::protocol::Protocol>::Error>
    for LocalMessagesHost
{
    async fn before_provider_request(
        &self,
        wire: litellm_host::event::WireRequest,
        _: litellm_host::event::RequestContext,
    ) -> Result<
        litellm_host::event::WireRequest,
        <Messages as litellm_host::protocol::Protocol>::Error,
    > {
        Ok(wire)
    }
    async fn on_event(
        &self,
        event: litellm_host::event::MachineEvent,
    ) -> Result<(), <Messages as litellm_host::protocol::Protocol>::Error> {
        litellm_host::lifecycle::CallObserver::observe(
            self,
            litellm_host::event::CallEvent::Machine(event),
        );
        Ok(())
    }
}
