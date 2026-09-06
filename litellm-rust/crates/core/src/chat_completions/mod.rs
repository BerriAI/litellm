//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

use crate::Error;
use crate::eligibility::native_route_decline;
use crate::request_context::LiteLlmRequestContext;
use crate::request_options::RequestOptions;
mod client;
mod common_utils;
pub mod conversation;
pub(crate) mod handler;
mod prepare;
pub mod response_utils;
pub mod transformation;
pub mod types;

use serde_json::{Map, Value};

use handler::execute_chat_completions_provider_call;
use prepare::{parse_messages, resolve_provider_config, resolve_request};
use transformation::{ChatCompletionsProviderConfig, Unsupported};
use types::{ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
    options: &RequestOptions,
    context: &LiteLlmRequestContext,
) -> Result<ChatCompletionsResponse, Error> {
    execute_chat_completions_provider_call(resolve_request(request, options.clone(), context)?)
        .await
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
    options: &RequestOptions,
    context: &LiteLlmRequestContext,
) -> Option<&'static str> {
    let Ok((_, provider, config)) = resolve_provider_config(model, custom_llm_provider) else {
        return Some("provider is not on the rust chat completions path");
    };
    let Ok(messages) = parse_messages(messages) else {
        return Some("unreadable message list");
    };
    if messages.is_empty() {
        return Some("empty message list");
    }
    unsupported_reason(
        provider,
        config,
        &messages,
        optional_params,
        options,
        context,
    )
    .map(|reason| reason.0)
}

fn unsupported_reason(
    provider: &str,
    config: &dyn ChatCompletionsProviderConfig,
    messages: &[ChatMessage],
    optional_params: &Map<String, Value>,
    options: &RequestOptions,
    context: &LiteLlmRequestContext,
) -> Option<Unsupported> {
    native_route_decline(true, &context.capabilities)
        .map(|reason| Unsupported(reason.reason()))
        .or_else(|| match provider {
            "anthropic" => options
                .anthropic
                .as_ref()
                .and_then(|anthropic| anthropic.user_id.as_ref())
                .map(|_| Unsupported("LiteLLM user metadata")),
            "bedrock" => options
                .bedrock
                .as_ref()
                .is_some_and(|bedrock| !bedrock.request_metadata_fields.is_empty())
                .then_some(Unsupported("LiteLLM request metadata forwarding")),
            _ => None,
        })
        .or_else(|| config.unsupported_reason(messages, optional_params))
}

#[cfg(test)]
mod tests;
