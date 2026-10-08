use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_http::{HttpSettings, Resolution};
use litellm_inference::Connection;
use litellm_inference_messages::{
    Error, MessagesCall, MessagesSettings, MessagesShaping,
    route::{Messages, MessagesMachine, MessagesOutput},
};
use litellm_inference_testing::RecordingSecrets;
use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};
use litellm_secrets::source::SecretSource;
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

fn body(value: Value) -> MessagesRequest {
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
        custom_llm_provider: Some("anthropic".into()),
        provider_specific_header: None,
        shaping: MessagesShaping::default(),
        connection: Connection {
            timeout: Some(Duration::from_secs(5)),
            ..Connection::default()
        },
    }
}

fn deployment<T>(value: T) -> Option<Sourced<T>> {
    Some(Sourced::new(value, InputSource::Deployment))
}

fn connected(call: MessagesCall, api_key: &str, api_base: String) -> MessagesCall {
    MessagesCall {
        connection: Connection {
            api_key: deployment(SecretValue::new(api_key)),
            api_base: deployment(api_base),
            ..call.connection
        },
        ..call
    }
}

fn with_timeout(call: MessagesCall, timeout: Duration) -> MessagesCall {
    MessagesCall {
        connection: Connection {
            timeout: Some(timeout),
            ..call.connection
        },
        ..call
    }
}

fn headers<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Option<Sourced<Map<String, Value>>> {
    deployment(
        pairs
            .into_iter()
            .map(|(name, value)| (name.to_string(), Value::from(value)))
            .collect(),
    )
}

fn machine(secrets: Arc<dyn SecretSource>) -> impl FnOnce(MessagesCall) -> MessagesMachine {
    move |request| messages_route(secrets).machine(request, None)
}

async fn run_with(
    secrets: Arc<RecordingSecrets>,
    call: MessagesCall,
) -> Result<MessagesOutput, Error> {
    let host = LocalMessagesHost::new(call);
    litellm_host_native::in_process::run_hosted(machine(secrets)(host.request()?), host.runtime())
        .await
}

/// Runs the route with a secret source that knows nothing, so no environment leaks in.
async fn run(call: MessagesCall) -> Result<MessagesOutput, Error> {
    run_with(Arc::new(RecordingSecrets::empty()), call).await
}

async fn run_message(call: MessagesCall) -> MessagesResponse {
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
    pub fn runtime(&self) -> litellm_host_native::in_process::Host<'_, (), Self, ()> {
        litellm_host_native::in_process::Host {
            services: &(),
            interceptors: self,
            stream: &(),
            observers: None,
        }
    }
}

impl litellm_host::lifecycle::CallObserver for LocalMessagesHost {
    fn observe(&self, _: litellm_host::lifecycle::CallEvent) {}
}
impl litellm_host::interceptors::Interceptors<<Messages as litellm_host::protocol::Protocol>::Error>
    for LocalMessagesHost
{
    async fn before_provider_request(
        &self,
        wire: litellm_host::interceptors::WireRequest,
        _: litellm_host::interceptors::RequestContext,
    ) -> Result<
        litellm_host::interceptors::WireRequest,
        <Messages as litellm_host::protocol::Protocol>::Error,
    > {
        Ok(wire)
    }
    async fn after_provider_response(
        &self,
        raw: litellm_host::interceptors::RawResponse,
    ) -> Result<(), <Messages as litellm_host::protocol::Protocol>::Error> {
        litellm_host::lifecycle::CallObserver::observe(
            self,
            litellm_host::lifecycle::CallEvent::Execution(
                litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
            ),
        );
        Ok(())
    }
}
