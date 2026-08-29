use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::{CoreError, CoreResult};
use crate::http_utils::{http_client, map_send_error, upstream_http};

use super::types::{AnthropicMessagesResponse, ProviderMessagesRequest};

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<AnthropicMessagesResponse> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder.send().await.map_err(map_send_error)?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::network(err.to_string()))?;

    if !status.is_success() {
        return Err(upstream_http(status, &text));
    }

    let response = serde_json::from_str(&text).map_err(|err| {
        CoreError::invalid_response(format!("invalid messages response JSON: {err}"))
    })?;
    // `transform_response` can only fail with an upstream error: the provider
    // already answered, so the request must not be retried blindly.
    Ok(request
        .config
        .transform_response(&request.model, response)?)
}

pub(super) async fn execute_messages_provider_stream(
    request: ProviderMessagesRequest,
) -> CoreResult<reqwest::Response> {
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(CoreError::invalid_request(
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

    let response = request_builder.send().await.map_err(map_send_error)?;
    let status = response.status();
    if !status.is_success() {
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::network(err.to_string()))?;
        return Err(upstream_http(status, &text));
    }
    Ok(response)
}
