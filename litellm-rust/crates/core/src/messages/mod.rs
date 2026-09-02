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
use crate::streaming::OpenedStream;

use handler::{execute_messages_provider_call, execute_messages_provider_stream};
use prepare::prepare_messages_call;
use types::{
    AnthropicMessagesResponse, MessagesRequest, MessagesStreamEvent, MessagesStreamRequest,
};

pub async fn messages(request: MessagesRequest<'_>) -> Result<AnthropicMessagesResponse, Error> {
    execute_messages_provider_call(prepare_messages_call(request)?).await
}

pub async fn messages_stream(request: MessagesRequest<'_>) -> Result<reqwest::Response, Error> {
    execute_messages_provider_stream(prepare_messages_call(request)?).await
}

pub async fn messages_event_stream(
    _request: MessagesStreamRequest,
) -> Result<OpenedStream<MessagesStreamEvent>, Error> {
    Err(crate::Error::Unsupported(
        "messages event streaming provider registration",
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
    async fn typed_event_stream_declines_until_a_provider_is_registered() {
        let body = serde_json::from_value(json!({
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "stream": true
        }))
        .expect("valid Messages stream request");
        let result = messages_event_stream(MessagesStreamRequest {
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
                "messages event streaming provider registration"
            ))
        ));
    }
}
