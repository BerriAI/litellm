use litellm_core::CoreResult;
use serde_json::Value;

mod client;
mod common_utils;
mod handler;
mod prepare;
mod types;

pub use types::MessagesRequest;

use handler::execute_messages_provider_call;
#[cfg(feature = "server")]
use handler::execute_messages_provider_stream;
use prepare::prepare_messages_call;

pub async fn messages(request: MessagesRequest<'_>) -> CoreResult<Value> {
    let prepared = prepare_messages_call(request)?;
    execute_messages_provider_call(prepared).await
}

#[cfg(feature = "server")]
pub(crate) async fn stream_messages(request: MessagesRequest<'_>) -> CoreResult<reqwest::Response> {
    let prepared = prepare_messages_call(request)?;
    execute_messages_provider_stream(prepared).await
}

#[cfg(test)]
mod tests;
