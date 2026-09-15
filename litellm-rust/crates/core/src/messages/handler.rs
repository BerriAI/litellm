use crate::call_lifecycle::provider::{ProviderHooks, ProviderRequest, ProviderResponse};
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
    hooks: &dyn ProviderHooks,
) -> Result<AnthropicMessagesResponse, Error> {
    let request = prepare_provider_request(request)?;
    let changed = hooks
        .before_request(ProviderRequest {
            model: request.model.clone(),
            url: request.url.clone(),
            headers: request.upstream_headers.clone(),
            body: request.body.clone(),
        })
        .await?;
    let request = super::types::ProviderMessagesRequest {
        model: changed.model,
        url: changed.url,
        upstream_headers: changed.headers,
        body: changed.body,
        ..request
    };
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

    let observed = hooks
        .after_response(ProviderResponse {
            status: status.as_u16(),
            body: text,
        })
        .await?;
    let observed_status = observed.status;
    let text = observed.body;
    if !(200..300).contains(&observed_status) {
        return Err(Error::Http {
            status: observed_status,
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
