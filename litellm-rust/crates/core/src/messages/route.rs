use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures_util::TryStreamExt;
use litellm_host::{
    event::PublicRequest,
    host::{Demand, Host, Verdict},
    machine::{CallMachine, HostChannel},
    protocol::Protocol,
};
use litellm_http::{Client, ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;
use litellm_types::llms::anthropic_messages::{
    anthropic_request::AnthropicMessagesRequest,
    anthropic_response::AnthropicMessagesResponse,
};
use serde_json::{Map, Value};

use super::{
    Error, MessagesCall, MessagesResponse, messages_body,
    handler::{self, execute},
    prepare::{prepare, resolve_provider},
};

pub const BODY_FIELDS: [&str; 22] = [
    "max_tokens",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "inference_geo",
    "top_p",
    "mcp_servers",
    "context_management",
    "compaction",
    "container",
    "output_format",
    "speed",
    "output_config",
    "cache_control",
    "reasoning_effort",
    "safeguards",
];

pub enum MessagesOutput {
    Message(Box<AnthropicMessagesResponse>),
    /// Every chunk already reached the host through `Deliver`.
    Streamed,
}

/// The upstream response as the caller sees it at stream hand-off, before any chunk.
pub struct MessagesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub struct Messages;

impl Protocol for Messages {
    type Response = MessagesOutput;
    type Error = Error;
    type Projection = MessagesCall;
    type Op = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesHost = HostChannel<Messages>;
pub type MessagesMachine = CallMachine<Messages>;

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
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.call
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .ok_or_else(|| Error::InvalidRequest("messages request was already projected".into()))
    }

    async fn open(&self, _: MessagesStreamHead) -> Result<Demand, Error> {
        Err(Error::Unsupported(
            "streamed responses need a streaming host",
        ))
    }

    async fn custom_op(&self, op: Infallible) -> Result<(), Error> {
        match op {}
    }
}

pub fn messages_machine(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesMachine, litellm_http::Error> {
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    let auth = resources.auth.clone();
    Ok(CallMachine::new(move |host| {
        Box::pin(drive(host, http, auth, secrets))
    }))
}

/// The call as its host sees it: projection first, then the same prepare and execute as
/// [`super::messages`], with each chunk of a stream handed over as it arrives.
async fn drive(
    host: MessagesHost,
    http: Client,
    auth: Arc<litellm_auth::AuthServices>,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesOutput, Error> {
    let call = host.project().await?;
    let caller_streams = call.body.params.stream == Some(true);
    let resolved = resolve_provider(&call.body.model, call.custom_llm_provider.as_deref())?;
    let body_map = |body: &AnthropicMessagesRequest| -> Result<Map<String, Value>, Error> {
        match serde_json::to_value(body).map_err(serialize_failure)? {
            Value::Object(map) => Ok(map),
            _ => unreachable!("a struct serializes to an object"),
        }
    };
    let params = host
        .pre_request(PublicRequest {
            model: call.body.model.clone(),
            custom_llm_provider: resolved.provider.as_str().to_string(),
            messages: serde_json::to_value(&call.body.messages).map_err(serialize_failure)?,
            params: body_map(&call.body)?
                .into_iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
                .collect(),
            fields: &BODY_FIELDS,
        })
        .await?;
    let patch = params
        .into_iter()
        .filter(|(name, _)| BODY_FIELDS.contains(&name.as_str()))
        .collect();
    let mut body = patched(&body_map(&call.body)?, patch);
    let mut recovered_thinking = false;
    loop {
        let request = prepare(
            MessagesCall {
                body: messages_body(body.clone())?,
                api_key: call.api_key.clone(),
                api_base: call.api_base.clone(),
                custom_llm_provider: call.custom_llm_provider.clone(),
                extra_headers: call.extra_headers.clone(),
                provider_specific_header: call.provider_specific_header.clone(),
                timeout: call.timeout,
                shaping: call.shaping.clone(),
            },
            secrets.as_ref(),
        )
        .await?;
        let provider = request.provider;
        let sent = serde_json::to_value(&request.body).map_err(serialize_failure)?;
        let response = match execute(&http, &auth, request, &host).await {
            Ok(response) => response,
            Err(error) => {
                if !recovered_thinking
                    && let Some(recovered) = handler::recover_thinking(&error, provider, &sent)?
                {
                    body = recovered;
                    recovered_thinking = true;
                    continue;
                }
                return Err(error);
            }
        };
        match response {
            MessagesResponse::Message(message) => {
                match host
                    .after_response(MessagesOutput::Message(message))
                    .await?
                {
                    Verdict::Return(MessagesOutput::Message(message)) if caller_streams => {
                        return handler::synthesize(&host, *message).await;
                    }
                    Verdict::Return(response) => return Ok(response),
                    Verdict::Resend(patch) => body = patched(&body, patch),
                }
            }
            MessagesResponse::Stream {
                headers,
                mut chunks,
            } => {
                if host.open(MessagesStreamHead { headers }).await? == Demand::Detached {
                    return Ok(MessagesOutput::Streamed);
                }
                while let Some(chunk) = chunks.try_next().await? {
                    if host.deliver(chunk).await? == Demand::Detached {
                        break;
                    }
                }
                return Ok(MessagesOutput::Streamed);
            }
        }
    }
}

fn patched(body: &Map<String, Value>, patch: Map<String, Value>) -> Map<String, Value> {
    body.iter()
        .filter(|(name, _)| !patch.contains_key(*name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .chain(
            patch
                .iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(name, value)| (name.clone(), value.clone())),
        )
        .collect()
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!(
        "failed to serialize Anthropic messages request: {err}"
    ))
}
