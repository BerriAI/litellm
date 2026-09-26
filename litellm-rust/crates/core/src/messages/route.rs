use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures_util::TryStreamExt;
use litellm_host::{
    host::{Demand, Host},
    machine::{CallMachine, HostChannel, MachineFault},
    protocol::Protocol,
};
use litellm_http::{Client, ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;

use super::{Error, MessagesCall, MessagesResponse, handler::execute, prepare::prepare};

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
    let request = prepare(call, secrets.as_ref()).await?;
    match execute(&http, &auth, request, &host).await? {
        MessagesResponse::Message(message) => Ok(MessagesOutput::Message(message)),
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
            Ok(MessagesOutput::Streamed)
        }
    }
}
