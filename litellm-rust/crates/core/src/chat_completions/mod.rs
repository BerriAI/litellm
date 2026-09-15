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
pub mod lifecycle;
mod prepare;
pub mod response_utils;
pub mod transformation;
pub mod types;

use serde_json::{Map, Value};

use prepare::{parse_messages, resolve_provider_config};
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    crate::call_lifecycle::provider::run_completed::<lifecycle::ChatCompletionsRoute>(
        request.into(),
    )
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

pub struct ChatCompletionsAdmission {
    pub model: String,
    pub provider: Option<String>,
    pub messages: Value,
    pub params: Map<String, Value>,
    pub headers: Option<Map<String, Value>>,
    pub context: AdmissionContext,
}

pub fn admit(
    inspection: crate::call_lifecycle::admission::Inspection<ChatCompletionsAdmission>,
) -> Result<(), crate::call_lifecycle::admission::AdmissionDecline> {
    use crate::call_lifecycle::admission::{AdmissionDecline, Inspection};
    let Inspection::Inspectable(admission) = inspection else {
        return Err(AdmissionDecline::Uninspectable);
    };
    let resolved = crate::routing_utils::provider::get_custom_llm_provider(
        &admission.model,
        admission.provider.as_deref(),
    );
    let provider = admission
        .provider
        .as_deref()
        .or_else(|| resolved.as_ref().map(|value| value.custom_llm_provider));
    if admission.context.stream {
        return Err(AdmissionDecline::Feature("streaming"));
    }
    if (provider == Some("anthropic") && admission.context.anthropic_user_id)
        || (provider == Some("bedrock") && admission.context.bedrock_metadata_owned)
    {
        return Err(AdmissionDecline::HostOperations);
    }
    #[cfg(feature = "bedrock-auth")]
    if provider == Some("bedrock")
        && admission.headers.as_ref().is_some_and(|headers| {
            headers
                .keys()
                .any(|name| crate::providers::bedrock::aws_base::is_sigv4_computed_header(name))
        })
    {
        return Err(AdmissionDecline::Feature(
            "request forwards a header AWS SigV4 computes",
        ));
    }
    let _ = admission.headers;
    match chat_completions_decline_reason(
        &admission.model,
        provider,
        admission.messages,
        &admission.params,
    ) {
        Some(reason) => Err(AdmissionDecline::Feature(reason)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
