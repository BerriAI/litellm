//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

use crate::Error;
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
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    execute_chat_completions_provider_call(resolve_request(request)?).await
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


#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionContext {
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub anthropic_user_id: bool,
    #[serde(default)]
    pub bedrock_metadata_owned: bool,
}

pub fn admit(
    model: &str,
    provider: Option<&str>,
    messages: Value,
    params: &Map<String, Value>,
    headers: Option<&Map<String, Value>>,
    context: AdmissionContext,
) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
    use crate::call_lifecycle::admission::AdmissionDecline;
    let resolved = crate::routing_utils::provider::get_custom_llm_provider(model, provider);
    let provider = provider.or_else(|| resolved.as_ref().map(|value| value.custom_llm_provider));
    if context.stream {
        return Err(AdmissionDecline::Feature("streaming"));
    }
    if (provider == Some("anthropic") && context.anthropic_user_id)
        || (provider == Some("bedrock") && context.bedrock_metadata_owned)
    {
        return Err(AdmissionDecline::HostOperations);
    }
    #[cfg(feature = "bedrock-auth")]
    if provider == Some("bedrock")
        && headers.is_some_and(|headers| {
            headers
                .keys()
                .any(|name| crate::providers::bedrock::aws_base::is_sigv4_computed_header(name))
        })
    {
        return Err(AdmissionDecline::Feature(
            "request forwards a header AWS SigV4 computes",
        ));
    }
    let _ = headers;
    match chat_completions_decline_reason(model, provider, messages, params) {
        Some(reason) => Err(AdmissionDecline::Feature(reason)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
