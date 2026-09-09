//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant.

use crate::Error;
pub mod common_utils;
mod handler;
pub mod lifecycle;
pub mod request;
pub mod transformation;
pub mod types;

use crate::lifecycle::StreamingCall;
pub use handler::execute_provider_messages_request;
pub(crate) use handler::{
    execute_messages_provider_call_with_transport, execute_messages_provider_stream_with_transport,
};
use types::{AnthropicMessagesResponse, MessagesRequest};

pub async fn messages(request: MessagesRequest) -> Result<AnthropicMessagesResponse, Error> {
    crate::runtime::LiteLlm::new().messages(request).await
}

pub async fn messages_stream(request: MessagesRequest) -> Result<StreamingCall, Error> {
    crate::runtime::LiteLlm::new()
        .messages_stream(request)
        .await
}
