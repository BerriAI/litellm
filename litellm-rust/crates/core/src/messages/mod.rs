//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`messages`] is the top-level entrypoint: give it a model, a body, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed non-streaming response. [`messages_stream`]
//! is the streaming variant.

use crate::Error;
pub mod common_utils;
mod handler;
pub mod lifecycle;
pub mod request;
mod streaming;
pub mod transformation;
pub mod types;

use crate::lifecycle::StreamingCall;
pub use handler::execute_provider_messages_request;
pub(crate) use handler::{
    execute_messages_provider_call_with_transport, execute_messages_provider_stream_with_transport,
};
use types::{AnthropicMessagesResponse, MessagesRequest};

pub async fn messages(request: MessagesRequest) -> Result<AnthropicMessagesResponse, Error> {
    crate::runtime::LiteLlm::new().messages(request).await
}

pub async fn messages_stream(request: MessagesRequest) -> Result<StreamingCall, Error> {
    crate::runtime::LiteLlm::new()
        .messages_stream(request)
        .await
}

pub async fn messages_stream_prepared<S>(
    request: types::ProviderMessagesRequest,
    context: crate::lifecycle::CallLifecycleContext,
    start_time: f64,
    services: std::sync::Arc<S>,
) -> Result<StreamingCall, Error>
where
    S: crate::lifecycle::Clock + crate::lifecycle::TerminalDispatcher + 'static,
{
    static TRANSPORT: std::sync::OnceLock<crate::runtime::NativeHttpTransport> =
        std::sync::OnceLock::new();
    let source = handler::execute_provider_messages_stream_with_transport(
        TRANSPORT.get_or_init(crate::runtime::NativeHttpTransport::new),
        request,
    )
    .await?;
    Ok(StreamingCall::new(
        source,
        Box::<streaming::AnthropicMessagesObserver>::default(),
        context,
        start_time,
        services,
    ))
}
