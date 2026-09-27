use std::{convert::Infallible, sync::Arc};

use bytes::Bytes;
use litellm_host::{
    call::{HostedCompletion, HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;

use super::{Error, MessagesCall};

pub type MessagesOutput = HostedCompletion<Box<AnthropicMessagesResponse>>;

/// The upstream response as the caller sees it at stream hand-off, before any chunk.
pub struct MessagesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub struct Messages;

impl Protocol for Messages {
    type Response = Box<AnthropicMessagesResponse>;
    type Error = Error;
    type Projection = MessagesCall;
    type Op = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesMachine = HostedMachine<Messages>;

pub fn messages_machine(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesMachine, litellm_http::Error> {
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    let auth = resources.auth.clone();
    Ok(hosted_call(move |call, host| async move {
        super::execute(&http, &auth, secrets.as_ref(), call, &host).await
    }))
}
