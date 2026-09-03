use super::types::IntoMessagesRequest;
use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::constants::MESSAGES_TIMEOUT_SECS;
use crate::error::Error;
use crate::http_utils::body::PreparedJsonBody;
use crate::http_utils::replay::{replay_client, send_json};
use std::time::Duration;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::prepare::prepare_provider_request;
use super::types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_messages_provider_call<B: IntoMessagesRequest>(
    request: MessagesRequest<'_, B>,
) -> Result<AnthropicMessagesResponse, Error> {
    let request = prepare_provider_request(request)?;
    let body = PreparedJsonBody::new(request.body.into_payload()?)?;
    let response = send_json(
        if body.is_streamed() {
            replay_client()?
        } else {
            http_client()
        },
        &request.url,
        &body,
        &request.upstream_headers,
        request
            .timeout
            .unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS)),
        None,
    )
    .await?;

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

    let response = serde_json::from_str(&text)
        .map_err(|err| Error::InvalidResponse(format!("invalid messages response JSON: {err}")))?;
    request.config.transform_response(&request.model, response)
}

pub(super) async fn execute_messages_provider_stream<B: IntoMessagesRequest>(
    request: MessagesRequest<'_, B>,
) -> Result<reqwest::Response, Error> {
    let request = prepare_provider_request(request)?;
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(Error::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let body = PreparedJsonBody::new(request.body.into_payload()?)?;
    let response = send_json(
        if body.is_streamed() {
            replay_client()?
        } else {
            http_client()
        },
        &request.url,
        &body,
        &request.upstream_headers,
        request
            .timeout
            .unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS)),
        None,
    )
    .await?;

    let status = response.status();
    if !status.is_success() {
        let text = response
            .text()
            .await
            .map_err(|err| Error::Network(err.to_string()))?;
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    Ok(response)
}
