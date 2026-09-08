use serde_json::Value;

use crate::error::Error;
use crate::http_utils::{http_request, truncate_error_body};

use super::client::http_client;
use super::request::build_provider_request;
use super::types::{
    ChatBodySnapshot, ChatCompletionsResponse, ChatEndpoint, ProviderChatCompletionsRequest,
    ProviderChatResponseData, ResolvedChatCompletionsRequest, SettledChatRequest,
};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_chat_completions_provider_call(
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    let request = build_provider_request(request)?;
    let body = serde_json::to_vec(&request.body).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize chat completions request: {err}"
        ))
    })?;
    let headers = signed_headers(&request, &body).await?;
    execute_settled_request(
        ChatBodySnapshot {
            endpoint: ChatEndpoint {
                model: request.model,
                config: request.config,
                url: request.url,
                timeout: request.timeout,
            },
            body,
        }
        .settle_headers(headers),
    )
    .await
}

pub(super) async fn execute_settled_request(
    request: SettledChatRequest,
) -> Result<ChatCompletionsResponse, Error> {
    let endpoint = request.snapshot.endpoint;
    let mut request_builder = http_client()
        .post(&endpoint.url)
        .body(request.snapshot.body);
    for (key, value) in &request.headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = endpoint.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder).await.map_err(|err| {
        // Failing to establish the connection means the request never went out,
        // so the host can still serve it. Everything else here, a timeout
        // above all, may have reached the provider and been answered.
        if err.is_connect() || err.is_builder() {
            Error::Connect(err.to_string())
        } else {
            Error::Network(err.to_string())
        }
    })?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        Error::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    endpoint
        .config
        .transform_response(&endpoint.model, ProviderChatResponseData { body })
        .map_err(as_response_error)
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// while building and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_) | Error::Http { .. }) => already,
        other => Error::InvalidResponse(other.to_string()),
    }
}

pub async fn signed_headers(
    request: &ProviderChatCompletionsRequest,
    body: &[u8],
) -> Result<Vec<(String, String)>, Error> {
    request.config.authorize(request, body).await
}
