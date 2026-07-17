use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use serde_json::Value;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::types::ProviderMessagesRequest;

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<Value> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid messages response JSON: {err}"))
    })?;
    let transformed = request
        .config
        .transform_response(&request.model, response)?;
    serde_json::to_value(transformed).map_err(|err| {
        CoreError::InvalidResponse(format!("failed to serialize messages response: {err}"))
    })
}
