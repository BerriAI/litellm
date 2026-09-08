use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::Error;
use crate::http_utils::http_request;
use crate::lifecycle::{StreamingMetadata, StreamingSource};

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::prepare::prepare_provider_request;
use super::types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_messages_provider_call(
    request: MessagesRequest,
) -> Result<AnthropicMessagesResponse, Error> {
    let request = prepare_provider_request(request)?;
    execute_prepared_messages_provider_call(request).await
}

pub async fn execute_prepared_messages_provider_call(
    request: super::types::ProviderMessagesRequest,
) -> Result<AnthropicMessagesResponse, Error> {
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
    request: MessagesRequest,
) -> Result<StreamingSource, Error> {
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
    let metadata = StreamingMetadata {
        status: status.as_u16(),
        content_type: response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
        cache_control: response
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
    };
    let stream = response.bytes_stream();
    Ok(StreamingSource {
        metadata,
        stream: Box::pin(futures_util::StreamExt::map(stream, |result| {
            result.map_err(|error| Error::Network(error.to_string()))
        })),
    })
}
