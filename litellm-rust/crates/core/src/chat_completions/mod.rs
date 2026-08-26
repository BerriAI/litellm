//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

mod client;
mod common_utils;
pub mod conversation;
pub(crate) mod handler;
mod prepare;
pub mod response_utils;
pub mod transformation;
pub mod types;

use serde_json::{Map, Value};

use crate::error::CoreResult;

use handler::execute_chat_completions_provider_call;
use prepare::{parse_messages, prepare_chat_completions_call, resolve_provider_config};
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> CoreResult<ChatCompletionsResponse> {
    execute_chat_completions_provider_call(prepare_chat_completions_call(request)?).await
}

/// Whether the core would accept this request, without resolving credentials or
/// touching the network.
///
/// A host that keeps the Python implementation asks this first so it can emit
/// its pre-call logging exactly once, on whichever path is about to run.
/// Returns the decline reason, or `None` when the request is accepted.
pub fn chat_completions_decline_reason(
    model: &str,
    custom_llm_provider: Option<&str>,
    messages: Value,
    optional_params: &Map<String, Value>,
) -> Option<&'static str> {
    let Ok((_, config)) = resolve_provider_config(model, custom_llm_provider) else {
        return Some("provider is not on the rust chat completions path");
    };
    let Ok(messages) = parse_messages(messages) else {
        return Some("unreadable message list");
    };
    if messages.is_empty() {
        return Some("empty message list");
    }
    config
        .unsupported_reason(&messages, optional_params)
        .map(|reason| reason.0)
}

#[cfg(test)]
mod tests;
