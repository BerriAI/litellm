//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant; it hands the raw upstream response back so a host
//! can splice the event stream to its own caller.

use crate::Error;
use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
mod client;
mod common_utils;
mod handler;
pub mod lifecycle;
mod prepare;
pub mod transformation;
pub mod types;

use handler::execute_messages_provider_stream;
use types::{AnthropicMessagesResponse, MessagesRequest};

pub async fn messages(request: MessagesRequest) -> Result<AnthropicMessagesResponse, Error> {
    let provider = request
        .custom_llm_provider
        .as_deref()
        .or_else(|| request.model.split_once('/').map(|(provider, _)| provider))
        .unwrap_or(ANTHROPIC_MESSAGES_PROVIDER);
    let context = crate::lifecycle::CallLifecycleContext::new(
        "messages",
        &request.model,
        provider,
        format!("{:032x}", rand::random::<u128>()),
    );
    lifecycle::messages(
        &lifecycle::NoopServices,
        request,
        lifecycle::Options::default(),
        context,
    )
    .await
    .into_result()
}

pub async fn messages_stream(request: MessagesRequest) -> Result<reqwest::Response, Error> {
    execute_messages_provider_stream(request).await
}

#[cfg(test)]
mod tests;
