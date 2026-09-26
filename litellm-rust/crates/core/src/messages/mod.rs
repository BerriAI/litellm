//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`route`] is the call as a machine a host drives, streaming or not. [`messages`] runs
//! it in process for a caller that already holds the request and wants the message.

pub mod types;
pub use crate::error::RouteError as Error;
mod common_utils;
mod handler;
mod prepare;
pub mod route;
use std::sync::Arc;

use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::source::EnvironmentSecrets;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use route::{LocalMessagesHost, MessagesCall, MessagesOutput, messages_machine};

pub async fn messages(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    call: MessagesCall,
) -> Result<AnthropicMessagesResponse, Error> {
    let secrets = Arc::new(EnvironmentSecrets::python_compatible(
        resources.pool.client(config, ClientVariant::Provider)?,
    ));
    match litellm_host::run::run(
        messages_machine(resources, config, secrets)?,
        &LocalMessagesHost::new(call),
    )
    .await?
    {
        MessagesOutput::Message(message) => Ok(*message),
        MessagesOutput::Streamed => Err(Error::Unsupported(
            "streamed responses need a streaming host",
        )),
    }
}
