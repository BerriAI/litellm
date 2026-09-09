use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::Error;
use crate::http_utils::{HttpClientProfile, http_client, http_request};
use crate::lifecycle::{StreamingMetadata, StreamingSource};
use futures_util::TryStreamExt;

use super::common_utils::truncate_error_body;
use super::request::{build_provider_request, build_provider_request_with_environment};
use super::types::{AnthropicMessagesResponse, MessagesRequest};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn execute_messages_provider_call(
    request: MessagesRequest,
) -> Result<AnthropicMessagesResponse, Error> {
    let request = build_provider_request(request)?;
    execute_provider_messages_request(request).await
}

pub(crate) async fn execute_messages_provider_call_with_transport(
    environment: &dyn crate::providers::auth::Environment,
    transport: &dyn crate::runtime::HttpTransport,
    request: MessagesRequest,
) -> Result<AnthropicMessagesResponse, Error> {
    let request =
        build_provider_request_with_environment(request, &|key| environment.environment(key))?;
    let super::types::ProviderMessagesRequest {
        model,
        config,
        url,
        http,
        timeout,
        ..
    } = request;
    let (body, headers) = http.into_parts();
    let response = transport
        .execute(crate::runtime::HttpRequest {
            method: reqwest::Method::POST,
            url,
            headers,
            body,
            timeout,
        })
        .await?;
    let text = String::from_utf8(response.body).map_err(|error| {
        Error::InvalidResponse(format!("invalid messages response body: {error}"))
    })?;
    if !(200..300).contains(&response.status) {
        return Err(Error::Http {
            status: response.status,
            body: truncate_error_body(&text),
        });
    }
    let response = serde_json::from_str(&text).map_err(|error| {
        Error::InvalidResponse(format!("invalid messages response JSON: {error}"))
    })?;
    config.transform_response(&model, response)
}

pub async fn execute_provider_messages_request(
    request: super::types::ProviderMessagesRequest,
) -> Result<AnthropicMessagesResponse, Error> {
    let super::types::ProviderMessagesRequest {
        model,
        config,
        url,
        http,
        timeout,
        ..
    } = request;
    let (body, headers) = http.into_parts();
    let client = http_client(HttpClientProfile::Standard)
        .map_err(|error| Error::Network(error.to_string()))?;
    let request_builder = headers
        .into_iter()
        .fold(client.post(&url).body(body), |builder, (key, value)| {
            builder.header(key, value)
        });
    let request_builder = match timeout {
        Some(timeout) => request_builder.timeout(timeout),
        None => request_builder,
    };

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
    config.transform_response(&model, response)
}

pub(crate) async fn execute_messages_provider_stream_with_transport(
    environment: &dyn crate::providers::auth::Environment,
    transport: &dyn crate::runtime::HttpTransport,
    request: MessagesRequest,
) -> Result<StreamingSource, Error> {
    let request =
        build_provider_request_with_environment(request, &|key| environment.environment(key))?;
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(Error::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let super::types::ProviderMessagesRequest {
        url, http, timeout, ..
    } = request;
    let (body, headers) = http.into_parts();
    let response = transport
        .execute_stream(crate::runtime::HttpRequest {
            method: reqwest::Method::POST,
            url,
            headers,
            body,
            timeout,
        })
        .await
        .map_err(|error| match error {
            Error::Connect(message) => Error::Network(message),
            error => error,
        })?;
    if !(200..300).contains(&response.status) {
        let body = response
            .stream
            .try_fold(Vec::new(), |mut body, chunk| async move {
                body.extend_from_slice(&chunk);
                Ok(body)
            })
            .await?;
        return Err(Error::Http {
            status: response.status,
            body: truncate_error_body(&String::from_utf8_lossy(&body)),
        });
    }
    Ok(StreamingSource {
        metadata: StreamingMetadata {
            status: response.status,
            content_type: response.content_type,
            cache_control: response.cache_control,
        },
        stream: response.stream,
    })
}
