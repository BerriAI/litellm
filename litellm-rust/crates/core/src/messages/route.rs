use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use futures_util::StreamExt;
use litellm_auth::SecretValue;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    host::{Demand, Host},
    machine::{CallMachine, HostChannel, MachineFault},
    protocol::Protocol,
};
use litellm_http::{Client, ClientVariant, HttpClientConfig};
use litellm_llms::base_llm::{
    anthropic_messages::streaming::{ByteStream, StreamDecoder, encode_anthropic_sse},
    auth::{Authenticated, resolve_auth},
};
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
    handler::{decode_response, network, provider_error, send},
    prepare::{invalid_request, prepare_provider_request, resolve_provider},
    types::MessagesShaping,
};

/// The caller's request as the host projects it.
pub struct MessagesCall {
    pub body: AnthropicMessagesRequest,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

/// Parses a caller's raw body, failing the way the route fails for any invalid request.
pub fn messages_body(body: Map<String, Value>) -> Result<AnthropicMessagesRequest, Error> {
    serde_json::from_value(Value::Object(body)).map_err(invalid_request)
}

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

impl From<MachineFault> for Error {
    fn from(fault: MachineFault) -> Self {
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "messages host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("messages {message}"),
        })
    }
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
        Box::pin(execute(host, http.clone(), auth.clone(), secrets.clone()))
    }))
}

async fn execute(
    host: MessagesHost,
    http: Client,
    auth: Arc<litellm_auth::AuthServices>,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesOutput, Error> {
    let call = host.project().await?;
    let resolved = resolve_provider(&call.body.model, call.custom_llm_provider.as_deref())?;
    let secrets = secrets
        .resolve(resolved.provider.config().secret_names())
        .await?;
    let api_key = call.api_key.clone().map(SecretValue::new);
    let request = prepare_provider_request(call, resolved, secrets.as_ref())?;
    let context = RequestContext {
        model: request.body.model.clone(),
        custom_llm_provider: request.provider.as_str().to_string(),
        optional_params: serde_json::to_value(&request.body.params).map_err(serialize_failure)?,
        secret_fields: Vec::new(),
        api_key,
    };
    let stream = request.body.params.stream == Some(true);
    let config = request.provider.config();
    let body = serde_json::to_value(&request.body).map_err(serialize_failure)?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let authenticated = resolve_auth(&auth, request.environment, &env_lookup).await?;
    let wire = host
        .before_send(
            WireRequest {
                url: request.url,
                headers: authenticated.headers,
                body,
            },
            context,
        )
        .await?;
    let response = send(
        &http,
        Authenticated {
            headers: wire.headers,
            signer: authenticated.signer,
        },
        &wire.url,
        &wire.body,
        request.timeout,
    )
    .await?;
    if !response.status().is_success() {
        return Err(provider_error(response).await);
    }
    if stream {
        return relay(&host, response, config.stream_decoder()).await;
    }
    let text = response.text().await.map_err(network)?;
    host.emit(MachineEvent::ResponseReceived {
        raw: RawResponse { body: text.clone() },
    })
    .await?;
    decode_response(config, &request.body.model, &text)
        .map(|message| MessagesOutput::Message(Box::new(message)))
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!(
        "failed to serialize Anthropic messages request: {err}"
    ))
}

/// Hands each upstream chunk to the caller as it arrives. A caller that stops reading
/// ends the upstream read, and the call completes with what it delivered.
///
/// A host on Anthropic SSE is relayed byte for byte. A host on another wire is decoded into
/// Anthropic stream events and re-encoded as Anthropic SSE.
async fn relay(
    host: &MessagesHost,
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
) -> Result<MessagesOutput, Error> {
    let head = MessagesStreamHead {
        headers: response
            .headers()
            .iter()
            .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
            .collect(),
    };
    if host.open(head).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    match decoder {
        None => relay_bytes(host, response).await,
        Some(decode) => relay_events(host, response, decode).await,
    }
}

async fn relay_bytes(
    host: &MessagesHost,
    mut response: reqwest::Response,
) -> Result<MessagesOutput, Error> {
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        if host.deliver(chunk).await? == Demand::Detached {
            break;
        }
    }
    Ok(MessagesOutput::Streamed)
}

async fn relay_events(
    host: &MessagesHost,
    response: reqwest::Response,
    decode: StreamDecoder,
) -> Result<MessagesOutput, Error> {
    let bytes: ByteStream = futures_util::stream::unfold(response, |mut response| async move {
        match response.chunk().await {
            Ok(Some(chunk)) => Some((Ok(chunk), response)),
            Ok(None) => None,
            Err(error) => Some((Err(std::io::Error::other(error)), response)),
        }
    })
    .boxed();
    let mut events = decode(bytes);
    while let Some(event) = events.next().await {
        let chunk = encode_anthropic_sse(&event?)?;
        if host.deliver(chunk).await? == Demand::Detached {
            break;
        }
    }
    Ok(MessagesOutput::Streamed)
}
