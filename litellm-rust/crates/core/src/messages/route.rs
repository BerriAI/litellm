use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedCompletion, HostedMachine, hosted_call},
    protocol::Protocol,
};
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
    type Request = MessagesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesMachine = HostedMachine<Messages>;

impl crate::CoreClient {
    pub fn messages_machine(
        &self,
    ) -> Result<
        impl FnOnce(super::MessagesCall) -> MessagesMachine + Send + Sync + use<>,
        litellm_http::Error,
    > {
        let http = self.provider_http()?;
        let auth = self.resources().auth.clone();
        let secrets = self.secret_source().clone();
        Ok(move |request| {
            hosted_call(request, move |call, _, hooks| async move {
                super::execute(&http, &auth, secrets.as_ref(), call, &hooks).await
            })
        })
    }
}
