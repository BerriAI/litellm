//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

mod error;
pub mod types;
pub use error::Error;
mod common_utils;
pub(crate) mod handler;
mod prepare;
use handler::execute_chat_completions_provider_call;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_types::utils::ChatCompletionsResponse;
use prepare::{parse_messages, resolve_provider_config, resolve_request};
use serde_json::{Map, Value};

use crate::chat_completions::types::ChatCompletionsRequest;

pub async fn chat_completions(
    pool: &HttpClientPool,
    config: &HttpClientConfig,
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    let request = resolve_request(request)?;
    let http = pool.client(config, ClientVariant::Provider)?;
    execute_chat_completions_provider_call(&http, request).await
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
