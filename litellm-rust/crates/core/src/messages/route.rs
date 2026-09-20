use std::{sync::Mutex, time::Duration};

use bytes::Bytes;
use litellm_auth::SecretValue;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    host::{Demand, Host},
    machine::{HostChannel, MachineFault, RouteMachine},
    route::Route,
};
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use serde_json::{Map, Value};

use super::{
    Error,
    common_utils::messages_provider_config,
    handler::{decode_response, network, provider_error, send},
    prepare::prepare_provider_request,
    types::MessagesRequest,
};
use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessagesOp {
    ProjectRequest,
}

pub enum MessagesOpResult {
    Request(Box<MessagesCall>),
}

/// The caller's request as the host projects it.
pub struct MessagesCall {
    pub cached_response: Option<Value>,
    pub options: crate::client::ClientOptions,
    pub model: String,
    pub body: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl MessagesCall {
    fn streams(&self) -> bool {
        self.body.get("stream").and_then(Value::as_bool) == Some(true)
    }
}

pub enum MessagesOutput {
    Message(Box<AnthropicMessagesResponse>),
    /// Every chunk already reached the host through `Deliver`.
    Streamed,
}

pub struct Messages;

impl Route for Messages {
    type Response = MessagesOutput;
    type Error = Error;
    type Op = MessagesOp;
    type OpResult = MessagesOpResult;
    type Chunk = Bytes;
    type StreamHead = ();
}

impl From<MachineFault> for Error {
    fn from(fault: MachineFault) -> Self {
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "messages host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("messages {message}"),
            MachineFault::Mismatch => "invalid messages host operation result".into(),
        })
    }
}

pub type MessagesHost = HostChannel<Messages>;
pub type MessagesMachine = RouteMachine<Messages>;

/// Whether this route serves the request, decided before any callback runs so a host
/// can still run its own path.
pub fn supports(model: &str, custom_llm_provider: Option<&str>, stream: bool) -> bool {
    let provider = get_custom_llm_provider(model, custom_llm_provider)
        .map(|resolved| resolved.custom_llm_provider)
        .or(custom_llm_provider);
    match provider {
        Some(ANTHROPIC_MESSAGES_PROVIDER) => true,
        Some(provider) => !stream && messages_provider_config(provider).is_some(),
        None => false,
    }
}

/// The in-process host for a request already in hand. It answers projection once and
/// observes nothing.
pub struct LocalMessagesHost {
    call: Mutex<Option<MessagesCall>>,
}

impl LocalMessagesHost {
    pub fn new(call: MessagesCall) -> Self {
        Self {
            call: Mutex::new(Some(call)),
        }
    }
}

impl Host<Messages> for LocalMessagesHost {
    async fn route(&self, op: MessagesOp) -> Result<MessagesOpResult, Error> {
        match op {
            MessagesOp::ProjectRequest => self
                .call
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .map(|call| MessagesOpResult::Request(Box::new(call)))
                .ok_or_else(|| {
                    Error::InvalidRequest("messages request was already projected".into())
                }),
        }
    }
}

pub fn messages_machine() -> MessagesMachine {
    RouteMachine::new(|host| Box::pin(execute(host)))
}

async fn execute(host: MessagesHost) -> Result<MessagesOutput, Error> {
    let MessagesOpResult::Request(mut call) = host.route(MessagesOp::ProjectRequest).await?;
    if let Some(budget) = &call.options.budget {
        budget.check()?;
    }
    let stream = call.streams();
    let cached = match call.cached_response.take() {
        Some(response) => Some(response),
        None => match &call.options.cache {
            Some(cache) => cache.get().await,
            None => None,
        },
    };
    if let Some(response) = cached {
        if !stream
            && let Ok(message) =
                serde_json::from_value::<AnthropicMessagesResponse>(response.clone())
        {
            call.options
                .record_messages_usage(message.usage.as_ref().unwrap_or(&Value::Null), true);
            return Ok(MessagesOutput::Message(Box::new(message)));
        }
        if stream
            && let Some(events) = response
                .get("litellm_cached_anthropic_sse_events")
                .and_then(Value::as_array)
            && events.iter().all(Value::is_string)
        {
            if host.open(()).await? == Demand::More {
                for event in events {
                    if host
                        .deliver(Bytes::from(event.as_str().unwrap_or_default().to_owned()))
                        .await?
                        == Demand::Detached
                    {
                        break;
                    }
                }
            }
            return Ok(MessagesOutput::Streamed);
        }
    }
    call.options.adjust_max_tokens(&mut call.body);
    let request = prepare_provider_request(MessagesRequest {
        model: &call.model,
        body: Value::Object(call.body.clone()),
        api_key: call.api_key.as_deref(),
        api_base: call.api_base.as_deref(),
        custom_llm_provider: call.custom_llm_provider.as_deref(),
        extra_headers: call.extra_headers.clone(),
        timeout: call.timeout,
    })?;
    if stream && request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(Error::Unsupported("streaming messages for this provider"));
    }
    let context = RequestContext {
        model: request.model.clone(),
        custom_llm_provider: request.provider.clone(),
        optional_params: Value::Object(
            call.body
                .iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        ),
        secret_fields: Vec::new(),
        api_key: call.api_key.clone().map(SecretValue::new),
    };
    let wire = host
        .before_send(
            WireRequest {
                url: request.url,
                headers: request.upstream_headers,
                body: request.body,
            },
            context,
        )
        .await?;
    let response = send(&wire.url, &wire.headers, &wire.body, request.timeout).await?;
    if !response.status().is_success() {
        return Err(provider_error(response).await);
    }
    if stream {
        return relay(&host, response, &call.options).await;
    }
    let text = response.text().await.map_err(network)?;
    host.emit(MachineEvent::ResponseReceived {
        raw: RawResponse { body: text.clone() },
    })
    .await?;
    let message = decode_response(request.config, &request.model, &text)?;
    call.options
        .record_messages_usage(message.usage.as_ref().unwrap_or(&Value::Null), false);
    if let Some(cache) = &call.options.cache
        && let Ok(value) = serde_json::to_value(&message)
    {
        cache.set(value).await;
    }
    Ok(MessagesOutput::Message(Box::new(message)))
}

/// Hands each upstream chunk to the caller as it arrives. A caller that stops reading
/// ends the upstream read, and the call completes with what it delivered.
async fn relay(
    host: &MessagesHost,
    mut response: reqwest::Response,
    options: &crate::client::ClientOptions,
) -> Result<MessagesOutput, Error> {
    if host.open(()).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    let mut collected = Vec::new();
    let mut detached = false;
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        collected.extend_from_slice(&chunk);
        if host.deliver(chunk).await? == Demand::Detached {
            detached = true;
            break;
        }
    }
    if let Some(stream) = litellm_core_utils::messages_stream::MessagesStream::parse(&collected) {
        options.record_messages_usage(&stream.usage, false);
        if !detached && let (Some(cache), Some(response)) = (&options.cache, stream.cached_response)
        {
            cache.set(response).await;
        }
    }
    Ok(MessagesOutput::Streamed)
}
