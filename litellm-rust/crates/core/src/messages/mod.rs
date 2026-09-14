//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant; it hands the raw upstream response back so a host
//! can splice the event stream to its own caller.

use crate::Error;
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


pub fn admit(
    model: &str,
    provider: Option<&str>,
    has_agentic_hook: bool,
) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
    use crate::call_lifecycle::admission::AdmissionDecline;
    let resolved = crate::routing_utils::provider::get_custom_llm_provider(model, provider);
    let provider = provider.or_else(|| resolved.as_ref().map(|value| value.custom_llm_provider));
    if provider
        .and_then(common_utils::messages_provider_config)
        .is_none()
    {
        return Err(AdmissionDecline::Provider);
    }
    if has_agentic_hook {
        return Err(AdmissionDecline::HostOperations);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
