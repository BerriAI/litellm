//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant; it hands the raw upstream response back so a host
//! can splice the event stream to its own caller.

use crate::Error;
use crate::request_context::LiteLlmRequestContext;
use crate::request_options::RequestOptions;
mod client;
mod common_utils;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use handler::{execute_messages_provider_call, execute_messages_provider_stream};
use types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn messages(
    request: MessagesRequest<'_>,
    options: &RequestOptions,
    _context: &LiteLlmRequestContext,
) -> Result<AnthropicMessagesResponse, Error> {
    execute_messages_provider_call(request, options.clone()).await
}

pub async fn messages_stream(
    request: MessagesRequest<'_>,
    options: &RequestOptions,
    _context: &LiteLlmRequestContext,
) -> Result<reqwest::Response, Error> {
    execute_messages_provider_stream(request, options.clone()).await
}

pub fn messages_provider_supported(provider: &str) -> bool {
    common_utils::messages_provider_config(provider).is_some()
}

#[cfg(test)]
mod tests;
