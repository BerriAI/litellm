//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant; it hands the raw upstream response back so a host
//! can splice the event stream to its own caller.

use crate::Error;
use crate::http_utils::SseFrameStream;
mod client;
mod common_utils;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use handler::{execute_messages_provider_call, execute_messages_provider_stream};
use types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn messages(request: MessagesRequest<'_>) -> Result<AnthropicMessagesResponse, Error> {
    execute_messages_provider_call(request).await
}

pub async fn messages_stream(request: MessagesRequest<'_>) -> Result<reqwest::Response, Error> {
    execute_messages_provider_stream(request).await
}

/// The streaming variant that yields complete server-sent-event frames.
///
/// Unlike [`messages_stream`], which hands back the raw upstream response for a
/// host to splice, this entrypoint keeps frame decoding in core and yields one
/// complete `event:`/`data:` frame per item, reassembled across network chunks.
pub async fn messages_stream_frames(request: MessagesRequest<'_>) -> Result<SseFrameStream, Error> {
    let response = execute_messages_provider_stream(request).await?;
    Ok(SseFrameStream::new(response))
}

#[cfg(test)]
mod tests;
