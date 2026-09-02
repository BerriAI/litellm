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

use crate::streaming::OpenedStream;
use serde_json::{Map, Value};

use handler::execute_chat_completions_provider_call;
use prepare::{parse_messages, prepare_chat_completions_call, resolve_provider_config};
use types::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatCompletionsStreamRequest, ChatStreamEvent,
};

pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
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

pub async fn chat_completions_stream(
    _request: ChatCompletionsStreamRequest,
) -> Result<OpenedStream<ChatStreamEvent>, Error> {
    Err(crate::Error::Unsupported(
        "chat completions streaming provider registration",
    ))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod stream_entrypoint_tests {
    use serde_json::json;

    use super::*;
    use crate::Error;
    use crate::streaming::{
        ProviderCredentials, StreamProviderId, StreamTarget, StreamTransportOptions,
    };

    #[tokio::test]
    async fn typed_stream_declines_until_a_provider_is_registered() {
        let body = serde_json::from_value(json!({
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .expect("valid chat stream request");
        let result = chat_completions_stream(ChatCompletionsStreamRequest {
            body,
            target: StreamTarget::new(
                StreamProviderId::Anthropic,
                ProviderCredentials::default(),
                None,
            ),
            transport: StreamTransportOptions::default(),
        })
        .await;

        assert!(matches!(
            result,
            Err(Error::Unsupported(
                "chat completions streaming provider registration"
            ))
        ));
    }
}
