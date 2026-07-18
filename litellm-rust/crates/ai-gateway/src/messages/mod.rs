use litellm_core::CoreResult;
use serde_json::Value;

mod client;
mod common_utils;
mod handler;
mod prepare;
mod types;

pub use types::MessagesRequest;

use handler::execute_messages_provider_call;
use prepare::prepare_messages_call;

pub async fn messages(request: MessagesRequest<'_>) -> CoreResult<Value> {
    let prepared = prepare_messages_call(request)?;
    execute_messages_provider_call(prepared).await
}

#[cfg(test)]
mod tests;
