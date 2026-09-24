use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use litellm_auth::SecretValue;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    host::{Demand, Host},
    machine::{HostChannel, MachineFault, RouteMachine},
    route::Route,
};
use litellm_llms::base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig;
use litellm_secrets::source::SecretSource;
use litellm_types::{
    llms::anthropic_messages::{
        anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
    },
    utils::ProviderSpecificHeaders,
};
use serde_json::{Map, Value};

use super::{
    Error,
    common_utils::messages_provider_config,
    handler::{decode_response, http_error, network, provider_error, send},
    prepare::{prepare_provider_request, resolve_provider},
    types::{MessagesRequest, MessagesShaping},
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
    pub model: String,
    pub body: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
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

pub fn messages_machine(secrets: Arc<dyn SecretSource>) -> MessagesMachine {
    RouteMachine::new(move |host| Box::pin(execute(host, secrets.clone())))
}

async fn execute(
    host: MessagesHost,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesOutput, Error> {
    let MessagesOpResult::Request(call) = host.route(MessagesOp::ProjectRequest).await?;
    let stream = call.streams();
    let resolved = resolve_provider(&call.model, call.custom_llm_provider.as_deref())?;
    let secrets = secrets.resolve(resolved.config.secret_names()).await?;
    let request = prepare_provider_request(
        MessagesRequest {
            model: &call.model,
            body: Value::Object(call.body.clone()),
            api_key: call.api_key.as_deref(),
            api_base: call.api_base.as_deref(),
            custom_llm_provider: call.custom_llm_provider.as_deref(),
            extra_headers: call.extra_headers.clone(),
            provider_specific_header: call.provider_specific_header.clone(),
            timeout: call.timeout,
            shaping: call.shaping.clone(),
        },
        resolved,
        secrets.as_ref(),
    )?;
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
    let response = if response.status().is_success() {
        response
    } else {
        resend_after_error(&host, request.config, &wire, request.timeout, response).await?
    };
    if stream {
        return relay(&host, response).await;
    }
    let text = response.text().await.map_err(network)?;
    host.emit(MachineEvent::ResponseReceived {
        raw: RawResponse { body: text.clone() },
    })
    .await?;
    decode_response(request.config, &request.model, &text)
        .map(|message| MessagesOutput::Message(Box::new(message)))
}

async fn resend_after_error(
    host: &MessagesHost,
    config: &dyn BaseAnthropicMessagesConfig,
    wire: &WireRequest,
    timeout: Option<Duration>,
    response: reqwest::Response,
) -> Result<reqwest::Response, Error> {
    let status = response.status().as_u16();
    let text = response.text().await.map_err(network)?;
    let Some(request) = serde_json::from_value::<AnthropicMessagesRequest>(wire.body.clone())
        .ok()
        .and_then(|request| config.request_after_http_error(status, &text, request))
    else {
        return Err(http_error(status, &text));
    };
    let body = serde_json::to_value(request)
        .map_err(|error| Error::InvalidRequest(format!("invalid messages request: {error}")))?;
    host.emit(MachineEvent::RequestResent { body: body.clone() })
        .await?;
    let response = send(&wire.url, &wire.headers, &body, timeout).await?;
    if response.status().is_success() {
        return Ok(response);
    }
    Err(provider_error(response).await)
}

/// Hands each upstream chunk to the caller as it arrives. A caller that stops reading
/// ends the upstream read, and the call completes with what it delivered.
async fn relay(
    host: &MessagesHost,
    mut response: reqwest::Response,
) -> Result<MessagesOutput, Error> {
    if host.open(()).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        if host.deliver(chunk).await? == Demand::Detached {
            break;
        }
    }
    Ok(MessagesOutput::Streamed)
}
