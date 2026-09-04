use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::Error;
use crate::http_utils::http_request;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::prepare::prepare_provider_request;
use super::types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_messages_provider_call(
    request: MessagesRequest<'_>,
) -> Result<AnthropicMessagesResponse, Error> {
    let request = prepare_provider_request(request)?;
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

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

pub(super) async fn execute_messages_provider_stream(
    request: MessagesRequest<'_>,
) -> Result<reqwest::Response, Error> {
    let request = prepare_provider_request(request)?;
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(Error::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;
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
