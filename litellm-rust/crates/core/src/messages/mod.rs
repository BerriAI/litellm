//! The Anthropic Messages call, the Rust equivalent of Python's
//! `litellm.messages()`.
//!
//! [`route`] is the call as a machine a host drives, streaming or not. [`messages`] runs
//! it in process for a caller that already holds the request and wants the message.

mod error;
pub mod types;
pub use error::Error;
mod common_utils;
mod handler;
mod prepare;
pub mod route;
use std::sync::Arc;

use litellm_http::{HttpClientConfig, HttpClientPool};
use litellm_secrets::source::EnvironmentSecrets;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use route::{LocalMessagesHost, MessagesCall, MessagesOutput, messages_machine};
use serde_json::Value;

use crate::messages::types::MessagesRequest;

pub async fn messages(
    pool: &HttpClientPool,
    config: &HttpClientConfig,
    request: MessagesRequest<'_>,
) -> Result<AnthropicMessagesResponse, Error> {
    let Value::Object(body) = request.body else {
        return Err(Error::InvalidRequest(
            "messages body must be an object".into(),
        ));
    };
    let call = MessagesCall {
        model: request.model.into(),
        body,
        api_key: request.api_key.map(Into::into),
        api_base: request.api_base.map(Into::into),
        custom_llm_provider: request.custom_llm_provider.map(Into::into),
        extra_headers: request.extra_headers,
        provider_specific_header: request.provider_specific_header,
        timeout: request.timeout,
        shaping: request.shaping,
    };
    let secrets = Arc::new(EnvironmentSecrets::python_compatible());
    match litellm_host::run::run(
        messages_machine(pool, config, secrets)?,
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
