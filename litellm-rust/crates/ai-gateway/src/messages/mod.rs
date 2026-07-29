use litellm_core::CoreResult;
use serde_json::Value;

mod client;
mod common_utils;
mod handler;
mod prepare;
mod types;

pub use types::MessagesRequest;

use handler::{execute_messages_provider_call, execute_messages_provider_stream};
use prepare::prepare_messages_call;

pub async fn messages(request: MessagesRequest<'_>) -> CoreResult<Value> {
    match execute_messages(request, false).await? {
        MessagesResponse::Json(body) => Ok(body),
        MessagesResponse::Stream { .. } => Err(litellm_core::CoreError::InvalidResponse(
            "non-streaming messages execution returned a stream".to_string(),
        )),
    }
}

pub(crate) enum MessagesResponse {
    Json(Value),
    #[allow(dead_code)]
    Stream {
        provider: String,
        response: reqwest::Response,
    },
}

pub(crate) async fn execute_messages(
    request: MessagesRequest<'_>,
    stream: bool,
) -> CoreResult<MessagesResponse> {
    let prepared = prepare_messages_call(request)?;
    if stream {
        let provider = prepared.provider.clone();
        execute_messages_provider_stream(prepared)
            .await
            .map(|response| MessagesResponse::Stream { provider, response })
    } else {
        execute_messages_provider_call(prepared)
            .await
            .map(MessagesResponse::Json)
    }
}

#[cfg(test)]
mod tests;
