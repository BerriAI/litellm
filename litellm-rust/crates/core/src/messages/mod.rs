//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant; it hands the raw upstream response back so a host
//! can splice the event stream to its own caller.

mod client;
mod common_utils;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use crate::error::CoreResult;

use handler::{execute_messages_provider_call, execute_messages_provider_stream};
use prepare::prepare_messages_call;
use types::{AnthropicMessagesResponse, MessagesRequest};

pub async fn messages(request: MessagesRequest<'_>) -> CoreResult<AnthropicMessagesResponse> {
    execute_messages_provider_call(prepare_messages_call(request)?).await
}

pub async fn messages_stream(request: MessagesRequest<'_>) -> CoreResult<reqwest::Response> {
    execute_messages_provider_stream(prepare_messages_call(request)?).await
}

#[cfg(test)]
mod tests;
