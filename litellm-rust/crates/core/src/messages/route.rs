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

impl super::MessagesRoute {
    pub fn machine(self, request: super::MessagesCall) -> MessagesMachine {
        hosted_call(request, move |call, _, hooks| async move {
            self.run(call, &hooks).await
        })
    }
}
