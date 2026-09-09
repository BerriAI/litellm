//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

use crate::Error;
mod common_utils;
pub mod conversation;
pub(crate) mod handler;
pub mod lifecycle;
pub mod request;
pub mod response_utils;
pub mod transformation;
pub mod types;

use serde_json::{Map, Value};

pub use handler::{as_response_error, signed_headers, signed_headers_with_services};
use request::resolve_provider_config;
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

use crate::lifecycle::{CallLifecycleContext, ExecutedCall};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    crate::runtime::LiteLlm::new()
        .chat_completions(request)
        .await
}

pub async fn execute_settled_with_terminal(
    request: types::SettledChatRequest,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    lifecycle::execute_settled(request, context).await
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
    messages: &[types::ChatMessage],
    optional_params: &Map<String, Value>,
) -> Option<&'static str> {
    let Ok((_, config)) = resolve_provider_config(model, custom_llm_provider) else {
        return Some("provider is not on the rust chat completions path");
    };
    if messages.is_empty() {
        return Some("empty message list");
    }
    config
        .unsupported_reason(messages, optional_params)
        .map(|reason| reason.0)
}
